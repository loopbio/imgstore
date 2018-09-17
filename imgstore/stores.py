# coding=utf-8
from __future__ import print_function, division, absolute_import
import re
import os.path
import itertools
import operator
import time
import logging
import glob
import uuid
import string
import datetime
import shutil
import json

import cv2
import yaml
import numpy as np
import pandas as pd
import pytz
import tzlocal
import dateutil.parser

try:
    import bloscpack
except ImportError:
    bloscpack = None

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'r+b')

from .util import ImageCodecProcessor, JsonCustomEncoder, FourCC, ensure_color, ensure_grayscale


STORE_MD_KEY = '__store'
STORE_MD_FILENAME = 'metadata.yaml'
STORE_LOCK_FILENAME = '.lock'

_VERBOSE_DEBUG_GETS = False
_VERBOSE_DEBUG_CHUNKS = False


def _extract_store_metadata(full_path):
    with open(full_path, 'r') as f:
        allmd = yaml.load(f)
    return allmd.pop(STORE_MD_KEY)


class _ImgStore(object):
    _version = 2
    _supported_modes = ''

    FRAME_MD = ('frame_number', 'frame_time')

    def __init__(self, basedir, mode, imgshape=None, imgdtype=None, chunksize=None, metadata=None,
                 encoding=None, write_encode_encoding=None):
        if mode not in self._supported_modes:
            raise ValueError('mode not supported')

        if imgdtype is not None:
            # ensure this is a string
            imgdtype = np.dtype(imgdtype).name

        self._basedir = basedir
        self._mode = mode
        self._imgshape = ()
        self._imgdtype = ''
        self._chunksize = 0
        self._encoding = None
        self._codec_proc = ImageCodecProcessor()  # used in read and write mode
        self._decode_image = None
        self._encode_image = None
        self._uuid = None

        self._metadata = {}
        self._user_metadata = {}

        self._tN = self._t0 = time.time()

        self._created_utc = self._timezone_local = None

        self.frame_min = np.nan
        self.frame_max = np.nan
        self.frame_number = np.nan
        self.frame_count = 0
        self.frame_time = np.nan

        self._log = logging.getLogger('imgstore')

        self._chunk_n = 0
        self._chunk_n_and_chunk_paths = ()

        # file pointer and filename of a file which can be used to store additional data per frame
        # (this is only created if data is actually stored)
        self._extra_data_fp = self._extra_data_fn = None

        if mode == 'w':
            if None in (imgshape, imgdtype, chunksize):
                raise ValueError('imgshape, imgdtype, chunksize must not be None')
            self._frame_n = 0
            self._init_write(imgshape, imgdtype, chunksize, metadata, encoding, write_encode_encoding)
        elif mode == 'r':
            # the index is a dict of tuples -> chunk_n
            # each tuple describes a contiguous series of framenumbers
            # e.g. (3,9) -> 5
            # means frames 3-9 are contained in chunk 5
            self._index = {}
            # for lookup by global index we also maintain a list of the chunk lengths
            self._index_chunklens = []  # [(chunk_n, chunk_len)]
            # the chunk index is a list of framenumbers, the index of the
            # framenumber is the position in the chunk
            self._chunk_index = []
            self._init_read()

            t0 = time.time()
            self._chunk_n_and_chunk_paths = self._find_chunks(chunk_numbers=None)
            self._log.debug('found %s chunks in in %fs' % (len(self._chunk_n_and_chunk_paths), time.time() - t0))

            self._build_index(self._chunk_n_and_chunk_paths)

            # reset to the start of the file and load the first chunk
            self._load_chunk(0)
            assert self._chunk_current_frame_idx == -1
            assert self._chunk_n == 0
            self.frame_number = np.nan  # we haven't read any frames yet

            # note: frame_idx always refers to the frame_idx within the chunk
            # whereas frame_index refers to the global frame_index from (0, frame_count]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_read(self):
        fullpath = os.path.join(self._basedir, STORE_MD_FILENAME)
        with open(fullpath, 'r') as f:
            allmd = yaml.load(f)
        smd = allmd.pop(STORE_MD_KEY)

        self._user_metadata.update(allmd)

        if smd['class'] == 'VideoImgStoreFFMPEG':
            store_class = 'VideoImgStore'
        else:
            store_class = smd['class']

        class_name = getattr(self, 'class_name', self.__class__.__name__)
        if store_class != class_name:
            raise ValueError('incompatible store')
        if smd['version'] != self._version:
            raise ValueError('incompatible store version')

        try:
            # noinspection PyShadowingNames
            uuid = smd['uuid']
        except KeyError:
            # noinspection PyShadowingNames
            uuid = None
        self._uuid = uuid

        self._imgshape = tuple(smd['imgshape'])
        self._imgdtype = smd['imgdtype']
        self._chunksize = int(smd['chunksize'])
        self._encoding = smd['encoding']
        self._metadata = smd

        # synthesize a created_date from old format stores
        if 'created_utc' not in smd:
            self._log.info('old store detected. synthesizing created datetime / timezone')

            dt = tz = None
            # we don't know the local timezone, so assume it is local
            if 'timezone' in allmd:
                # noinspection PyBroadException
                try:
                    tz = pytz.timezone(allmd['timezone'])
                except Exception:
                    pass
            if tz is None:
                tz = tzlocal.get_localzone()

            # first the filename
            m = re.match(r"""(.*)(20[\d]{6}_\d{6}).*""", os.path.basename(self._basedir))
            if m:
                name, datestr = m.groups()
                # ive always been careful to make the files named with the local time
                time_tuple = time.strptime(datestr, '%Y%m%d_%H%M%S')
                _dt = datetime.datetime(*(time_tuple[0:6]))
                dt = tz.localize(_dt).astimezone(pytz.utc)

            # then the modification time of the file
            if dt is None:
                # file modifications are local time
                ts = os.path.getmtime(fullpath)
                dt = datetime.datetime.fromtimestamp(ts, tz=tzlocal.get_localzone()).astimezone(pytz.utc)

            self._created_utc = dt
            self._timezone_local = tz
        else:
            # ensure that created_utc always has the pytz.utc timezone object because fuck you python
            # and fuck you dateutil for having different UTC tz objects
            # https://github.com/dateutil/dateutil/issues/131
            _dt = dateutil.parser.parse(smd['created_utc'])
            self._log.debug('parsed created_utc: %s (from %r)' % (_dt.isoformat(), _dt))
            try:
                self._created_utc = _dt.astimezone(pytz.utc)  # aware object can be in any timezone
            except ValueError:  # naive
                self._created_utc = _dt.replace(tzinfo=pytz.utc)  # d must be in UTC
            self._timezone_local = pytz.timezone(smd['timezone_local'])

        # if encoding is unset, autoconvert is no-op
        self._codec_proc.set_default_code(self._encoding)
        self._decode_image = self._codec_proc.autoconvert

    def _init_write(self, imgshape, imgdtype, chunksize, metadata, encoding, write_encode_encoding):
        for e in (encoding, write_encode_encoding):
            if e:
                if not self._codec_proc.check_code(e):
                    raise ValueError('unsupported store image encoding: %s' % e)

        # if encoding is unset, autoconvert is no-op
        self._codec_proc.set_default_code(write_encode_encoding)
        self._encode_image = self._codec_proc.autoconvert

        if write_encode_encoding:
            # as we always encode to color
            imgshape = [imgshape[0], imgshape[1], 3]

        if not os.path.exists(self._basedir):
            os.makedirs(self._basedir)

        self._imgshape = imgshape
        self._imgdtype = imgdtype
        self._chunksize = chunksize

        self._uuid = uuid.uuid4().hex
        # because fuck you python that utcnow is naieve. kind of fixed in python >3.2
        self._created_utc = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        self._timezone_local = tzlocal.get_localzone()

        store_md = {'imgshape': imgshape,
                    'imgdtype': self._imgdtype,
                    'chunksize': chunksize,
                    'class': self.__class__.__name__,
                    'version': self._version,
                    'encoding': encoding,
                    # actually write the string as naieve because we have guarenteed it is UTC
                    'created_utc': self._created_utc.replace(tzinfo=None).isoformat(),
                    'timezone_local': str(self._timezone_local),
                    'uuid': self._uuid}

        if metadata is None:
            metadata = {STORE_MD_KEY: store_md}
        elif isinstance(metadata, dict):
            try:
                metadata[STORE_MD_KEY].update(store_md)
            except KeyError:
                metadata[STORE_MD_KEY] = store_md
        else:
            raise ValueError('metadata must be a dictionary')

        with open(os.path.join(self._basedir, STORE_MD_FILENAME), 'wt') as f:
            yaml.safe_dump(metadata, f)

        with open(os.path.join(self._basedir, STORE_LOCK_FILENAME), 'a') as f:
            pass

        smd = metadata.pop(STORE_MD_KEY)
        self._metadata = smd
        self._user_metadata.update(metadata)

        self._save_chunk(None, self._chunk_n)

    def _save_chunk_metadata(self, path_without_extension, extension='.npz'):
        path = path_without_extension + extension
        self._save_index(path, self._chunk_md)

        # also calculate the filename of the extra file to hold any data
        if self._extra_data_fp is not None:
            self._extra_data_fp.write(']')
            self._extra_data_fp.close()
            self._extra_data_fp = None

    def _new_chunk_metadata(self, chunk_path):
        self._extra_data_fn = chunk_path + '.extra.json'
        self._chunk_md = {k: [] for k in _ImgStore.FRAME_MD}
        self._chunk_md.update(self._metadata)

    def _save_image_metadata(self, frame_number, frame_time):
        self._chunk_md['frame_number'].append(frame_number)
        self._chunk_md['frame_time'].append(frame_time)

    @classmethod
    def supported_formats(cls):
        raise NotImplementedError

    @classmethod
    def supports_format(cls, fmt):
        return fmt in cls.supported_formats()

    @staticmethod
    def _save_index(path_with_extension, data_dict):
        _, extension = os.path.splitext(path_with_extension)
        with open(path_with_extension, 'w') as f:
            if extension == '.yaml':
                yaml.safe_dump(data_dict, f)
            elif extension == '.npz':
                # noinspection PyTypeChecker
                np.savez(f, **data_dict)
            else:
                raise ValueError('unknown index format: %s' % extension)

    @staticmethod
    def _remove_index(path_without_extension):
        for extension in ('.npz', '.yaml'):
            path = path_without_extension + extension
            if os.path.exists(path):
                os.unlink(path)

    @staticmethod
    def _load_index(path_without_extension):
        for extension in ('.npz', '.yaml'):
            path = path_without_extension + extension
            if os.path.exists(path):
                with open(path, 'r') as f:
                    if extension == '.yaml':
                        dat = yaml.safe_load(f)
                        return {k: dat[k] for k in _ImgStore.FRAME_MD}
                    elif extension == '.npz':
                        dat = np.load(f)
                        return {k: dat[k].tolist() for k in _ImgStore.FRAME_MD}

        raise IOError('could not find index %s' % path_without_extension)

    def _build_index(self, chunk_n_and_chunk_paths):
        t0 = time.time()
        for chunk_n, chunk_path in sorted(chunk_n_and_chunk_paths, key=operator.itemgetter(0)):
            try:
                idx = self._load_index(chunk_path)
            except IOError:
                self._log.warning('missing index for chunk %s' % chunk_n)
                continue

            if not idx['frame_number']:
                # empty chunk
                continue

            chunk_len = len(idx['frame_number'])
            self.frame_count += chunk_len
            self._t0 = min(self._t0, np.min(idx['frame_time']))
            self._tN = max(self._tN, np.max(idx['frame_time']))

            for frame_range in _extract_ranges(idx['frame_number']):
                self._index[frame_range] = chunk_n
                if _VERBOSE_DEBUG_CHUNKS:
                    self._log.debug('index:framenumbers chunk: %d holds:%r' % (chunk_n, frame_range))
            self._index_chunklens.append((chunk_n, chunk_len))

        if _VERBOSE_DEBUG_CHUNKS:
            _chunk_range = xrange(0, -2, -1)
            for _chunk_n, _chunk_len in self._index_chunklens:
                _chunk_range = xrange(_chunk_range[-1] + 1, _chunk_range[-1] + 1 + _chunk_len)
                self._log.debug('index:index chunk: %d holds:%r' % (_chunk_n, list(_chunk_range)))

        self._log.debug('built index in %fs' % (time.time() - t0))

        k = self._index.keys()
        self.frame_min = np.min(k)
        self.frame_max = np.max(k)

        self._log.debug('frame range %f -> %f' % (self.frame_min, self.frame_max))

    def _load_chunk_metadata(self, path_without_extension):
        self._chunk_md = self._load_index(path_without_extension)

    def get_frame_metadata(self):
        dat = {k: [] for k in _ImgStore.FRAME_MD}
        for chunk_n, chunk_path in self._find_chunks(self.chunks):
            idx = self._load_index(chunk_path)
            for k in _ImgStore.FRAME_MD:
                dat[k].extend(idx[k])
        return dat

    @property
    def has_extra_data(self):
        for chunk_n, chunk_path in self._chunk_n_and_chunk_paths:
            path = chunk_path + '.extra.json'
            if os.path.exists(path):
                return True
        return False

    def get_extra_data(self):
        dfs = []
        for chunk_n, chunk_path in self._chunk_n_and_chunk_paths:
            path = chunk_path + '.extra.json'
            if os.path.exists(path):
                dfs.append(pd.read_json(path, orient='record'))
        return pd.concat(dfs, axis=0, ignore_index=True)

    def add_extra_data(self, **data):
        if not data:
            return

        data['frame_time'] = self.frame_time
        data['frame_number'] = self.frame_number
        data['frame_index'] = self._frame_n - 1  # we post-increment in add_frame

        # noinspection PyBroadException
        try:
            txt = json.dumps(data, cls=JsonCustomEncoder)
        except Exception:
            self._log.warning('error writing extra data', exc_info=True)
            return

        if self._extra_data_fp is None:
            self._extra_data_fp = open(self._extra_data_fn, 'w')
            self._extra_data_fp.write('[')
        else:
            self._extra_data_fp.write(', ')
        self._extra_data_fp.write(txt)

    def _save_image(self, img, frame_number, frame_time):  # pragma: no cover
        raise NotImplementedError

    def _save_chunk(self, old, new):  # pragma: no cover
        raise NotImplementedError

    def _load_image(self, idx):  # pragma: no cover
        raise NotImplementedError

    def _load_chunk(self, n):  # pragma: no cover
        raise NotImplementedError

    def _find_chunks(self, chunk_numbers):  # pragma: no cover
        raise NotImplementedError

    def __len__(self):
        return self.frame_count

    @property
    def created(self):
        return self._created_utc, self._timezone_local

    @property
    def uuid(self):
        return self._uuid

    @property
    def chunks(self):
        return sorted(set(self._index.values()))

    @property
    def user_metadata(self):
        return self._user_metadata

    @property
    def filename(self):
        return self._basedir

    @property
    def full_path(self):
        return os.path.join(self._basedir, STORE_MD_FILENAME)

    @property
    def image_shape(self):
        # if encoding is specified, we always decode to bgr (color)
        if self._encoding:
            return self._imgshape[0], self._imgshape[1], 3
        else:
            return self._imgshape

    @property
    def duration(self):
        return self._tN - self._t0

    @staticmethod
    def _extract_only_frame(basedir, chunk_n, frame_n, smd):
        raise NotImplementedError

    @classmethod
    def extract_only_frame(cls, full_path, frame_index, _smd=None):
        if _smd is None:
            smd = _extract_store_metadata(full_path)
        else:
            smd = _smd

        chunksize = int(smd['chunksize'])

        # go directly to the chunk
        chunk_n = frame_index // chunksize
        frame_n = frame_index % chunksize

        return cls._extract_only_frame(basedir=os.path.dirname(full_path),
                                       chunk_n=chunk_n,
                                       frame_n=frame_n,
                                       smd=smd)

    def disable_decoding(self):
        self._decode_image = lambda x: x

    def add_image(self, img, frame_number, frame_time):
        self._save_image(self._encode_image(img), frame_number, frame_time)

        self.frame_max = np.nanmax((frame_number, self.frame_max))
        self.frame_min = np.nanmin((frame_number, self.frame_min))
        self.frame_number = frame_number
        self.frame_time = frame_time

        if self._frame_n == 0:
            self._t0 = frame_time
        self._tN = frame_time

        self._frame_n += 1
        if (self._frame_n % self._chunksize) == 0:
            old = self._chunk_n
            new = self._chunk_n + 1
            self._save_chunk(old, new)
            self._chunk_n = new

        self.frame_count = self._frame_n

    def _get_next_framenumber_and_chunk_frame_idx(self):
        if self.frame_number == self.frame_max:
            raise EOFError

        idx = self._chunk_current_frame_idx + 1
        try:
            frame_number = self._chunk_index[idx]
        except IndexError:
            # open the next chunk
            self._load_chunk(self._chunk_n + 1)
            # first frame is start of chunk
            idx = 0
            frame_number = self._chunk_index[idx]

        return frame_number, idx

    def get_next_framenumber(self):
        return self._get_next_framenumber_and_chunk_frame_idx()[0]

    def get_next_image(self):
        frame_number, idx = self._get_next_framenumber_and_chunk_frame_idx()
        if _VERBOSE_DEBUG_GETS:
            self._log.debug('get_next_image frame_number: %s idx %s' % (frame_number, idx))
        return self._get_image_by_frame_number(frame_number, exact_only=True, frame_idx=idx)

    def _get_image_by_frame_index(self, frame_index):
        """
        return the frame at the following index in the store
        """
        if frame_index < 0:
            raise ValueError('seeking to negative index not supported')

        if _VERBOSE_DEBUG_CHUNKS:
            self._log.debug('seek by frame_index %s' % frame_index)

        # go through the global index and find where our index is
        chunk_n = frame_idx = -1

        _chunk_range = xrange(0, -2, -1)
        for _chunk_n, _chunk_len in self._index_chunklens:
            _chunk_range = xrange(_chunk_range[-1] + 1, _chunk_range[-1] + 1 + _chunk_len)
            if frame_index in _chunk_range:
                chunk_n = _chunk_n
                # make chunk relative
                frame_idx = frame_index - _chunk_range[0]
                break

        if chunk_n == -1:
            raise ValueError('frame_index %s not found in index' % frame_idx)

        # reset to start of chunk for load_image
        self._load_chunk(chunk_n)

        self._log.debug('seek found in chunk %d attempt read chunk index %d' % (self._chunk_n, frame_idx))

        # ensure the read works before setting frame_number
        _img, (_frame_number, _frame_timestamp) = self._load_image(frame_idx)
        img = self._decode_image(_img)

        self._chunk_current_frame_idx = frame_idx
        self.frame_number = _frame_number

        return img, (_frame_number, _frame_timestamp)

    def _get_image_by_frame_number(self, frame_number, exact_only, frame_idx):
        # there is a high likelihood the current chunk holds the next frame
        # so look there first

        if _VERBOSE_DEBUG_CHUNKS:
            self._log.debug('seek by frame_number %s (exact: %s) frame_idx %s' % (frame_number, exact_only, frame_idx))

        if frame_idx is None:
            try:
                frame_idx = self._chunk_index.index(frame_number)
            except ValueError:
                frame_idx = None

        if frame_idx is None:
            chunk_n = -1
            found = False
            for fn_range, chunk_n in self._index.items():
                if (frame_number >= fn_range[0]) and (frame_number <= fn_range[1]):
                    found = True
                    break

            if not found:
                if exact_only:
                    raise ValueError('frame #%s not found in any chunk' % frame_number)
                else:
                    # iterate the index again! and find the nearest frame
                    fns = np.array(list(self._index.keys())).flatten()
                    chunks = list(self._index.values())

                    # find distance between desired frame, and the nearest chunk range
                    diffs = np.abs(fns - frame_number)
                    # find the index of the minimum difference
                    _fn_index = np.argmin(diffs)
                    # consecutive elements of the flattened range array refer to the same
                    _chunk_idx = _fn_index // 2

                    orig_frame_number = frame_number

                    # the closest chunk
                    # noinspection PyTypeChecker
                    chunk_n = chunks[_chunk_idx]
                    frame_number = fns[_fn_index]
                    self._log.debug("closest frame to %s is %s (chunk %s)" % (orig_frame_number, frame_number, chunk_n))

            self._load_chunk(chunk_n)
            try:
                frame_idx = self._chunk_index.index(frame_number)
            except ValueError:
                raise ValueError('%s %s not found in chunk %s' % ('frame_number', frame_number, chunk_n))

        if _VERBOSE_DEBUG_CHUNKS:
            self._log.debug('seek found in chunk %d attempt read chunk index %d' % (self._chunk_n, frame_idx))

        # ensure the read works before setting frame_number
        _img, (_frame_number, _frame_timestamp) = self._load_image(frame_idx)
        img = self._decode_image(_img)

        self._chunk_current_frame_idx = frame_idx
        self.frame_number = _frame_number

        return img, (_frame_number, _frame_timestamp)

    def get_image(self, frame_number, exact_only=True, frame_index=None):
        """
        seek to the supplied frame_number or frame_index. If frame_index is supplied get that image,
        otherwise get the image corresponding to frame_number

        :param frame_number:  (frame_min, frame_max)
        :param exact_only: If False return the nearest frame
        :param frame_index: frame_index (0, frame_count]
        """
        if _VERBOSE_DEBUG_GETS:
            self._log.debug('get_image %s (exact: %s) frame_idx %s' % (frame_number, exact_only, frame_index))
        if frame_index is not None:
            return self._get_image_by_frame_index(frame_index)
        else:
            return self._get_image_by_frame_number(frame_number, exact_only=exact_only, frame_idx=None)

    def close(self):
        if self._mode in 'wa':
            self._save_chunk(self._chunk_n, None)
            try:
                if os.path.isfile(os.path.join(self._basedir, STORE_LOCK_FILENAME)):
                    os.remove(os.path.join(self._basedir, STORE_LOCK_FILENAME))
            except OSError:
                self._log.warn('could not remove lock file', exc_info=True)
            except Exception:
                self._log.warn('could not remove lock file (unknown error)', exc_info=True)

    def empty(self):
        if self._mode != 'w':
            raise ValueError('can only empty stores for writing')

        self.close()

        self._tN = self._t0 = time.time()

        self.frame_min = np.nan
        self.frame_max = np.nan
        self.frame_number = np.nan
        self.frame_count = 0
        self.frame_time = np.nan

        self._chunk_n = 0
        self._chunk_n_and_chunk_paths = ()

        if self._extra_data_fp is not None:
            self._extra_data_fp.close()
            os.unlink(self._extra_data_fn)
            self._extra_data_fp = self._extra_data_fn = None

        self._frame_n = 0

    def reindex(self):
        """ modifies the current imgstore so that all framenumbers before frame_number=0 are negative

        if there are multiple frame_numbers equal to zero then this operation aborts. this functions also
        updates the frame_number of any stored extra data. the original frame_number prior to calling
        reindex() is stored in '_frame_number_before_reindex' """
        md = self.get_frame_metadata()
        fn = md['frame_number']

        nzeros = fn.count(0)
        if nzeros != 1:
            raise ValueError("%d frame_number=0 found (should be 1)" % nzeros)

        # get index and time of sync frame
        zero_idx = fn.index(0)
        self._log.info('reindexing about frame_number=0 at index=%d' % zero_idx)

        fn_new = fn[:]
        fn_new[:zero_idx] = range(-zero_idx, 0)

        for chunk_n, chunk_path in sorted(self._find_chunks(chunk_numbers=None), key=operator.itemgetter(0)):
            # noinspection PyUnresolvedReferences
            ind = self._load_index(chunk_path)

            ofn = ind['frame_number'][:]
            nfn = fn_new[chunk_n*self._chunksize:(chunk_n*self._chunksize) + self._chunksize]
            assert len(ofn) == len(nfn)

            new_ind = {'frame_time': ind['frame_time'],
                       'frame_number': nfn,
                       '_frame_number_before_reindex': ofn}

            # noinspection PyUnresolvedReferences
            self._remove_index(chunk_path)
            self._save_index(chunk_path + '.npz', new_ind)

            self._log.debug('reindexed chunk %d (%s)' % (chunk_n, chunk_path))

            ed_path = chunk_path + '.extra.json'
            if os.path.exists(ed_path):
                with open(ed_path, 'r') as f:
                    ed = json.load(f)

                # noinspection PyBroadException
                try:
                    df = pd.DataFrame(ed)
                    if 'frame_index' not in df.columns:
                        raise ValueError('can not reindex extra-data on old format stores')
                    df['_frame_number_before_reindex'] = df['frame_number']
                    df['frame_number'] = df.apply(lambda r: fn_new[int(r.frame_index)], axis=1)
                    with open(ed_path, 'w') as f:
                        df.to_json(f, orient='records')
                    self._log.debug('reindexed chunk %d metadata (%s)' % (chunk_n, ed_path))
                except Exception:
                    self._log.error('could not update chunk extra data to new framenumber', exc_info=True)

    @staticmethod
    def _extract_only_frame(basedir, chunk_n, frame_n, smd):
        return None


def _extract_ranges(data):
    # convert a list of integers into a list of contiguous ranges
    # [2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 20] -> [(2,5), (12,17), (20, 20)]
    # http://stackoverflow.com/a/2154437
    ranges = []
    for key, group in itertools.groupby(enumerate(sorted(data)), lambda x: x[0] - x[1]):
        group = list(map(operator.itemgetter(1), group))
        if len(group) > 1:
            ranges.append((group[0], group[-1]))
        else:
            ranges.append((group[0], group[0]))

    return ranges


class DirectoryImgStore(_ImgStore):
    _supported_modes = 'wr'

    _cv2_fmts = {'tif', 'png', 'jpg', 'ppm', 'pgm', 'bmp'}
    _raw_fmts = {'npy', 'bpk'}

    def __init__(self, **kwargs):

        self._chunk_cdir = ''
        self._chunk_md = {}

        # keep compat with VideoImgStore
        kwargs.pop('videosize', None)
        # keep compat with VideoImgStoreFFMPEG
        kwargs.pop('seek', None)

        self._format = kwargs.pop('format', None)
        if kwargs['mode'] == 'w':
            if 'chunksize' not in kwargs:
                kwargs['chunksize'] = 100
            if self._format is None:
                raise ValueError('image format must be supplied')
            metadata = kwargs.get('metadata', {})
            metadata[STORE_MD_KEY] = {'format': self._format}
            kwargs['metadata'] = metadata
            kwargs['encoding'] = kwargs.pop('encoding', None)

        _ImgStore.__init__(self, **kwargs)

        if self._mode == 'r':
            self._format = self._metadata['format']

        self._color = (self._imgshape[-1] == 3) & (len(self._imgshape) == 3)

        if (self._mode == 'w') and (self._format == 'pgm') and self._color:
            self._log.warn("store created with color image shape but using grayscale 'pgm' format")

        if self._format not in itertools.chain(self._cv2_fmts, ('npy', 'bpk')):
            raise ValueError('unknown format %s' % self._format)

        if (self._format == 'bpk') and (bloscpack is None):
            raise ValueError('bloscpack not installed or available')

        if (self._format == 'npy') and (np.__version__ < '1.9.0') and (self._mode in 'wa'):
            # writing to npy takes an unecessary copy in memory [1], which was fixed in version 1.9.0
            # [1] https://www.youtube.com/watch?v=TZdqeEd7iTM
            pass

    def _save_image(self, img, frame_number, frame_time):
        dest = os.path.join(self._chunk_cdir, '%06d.%s' % (self._frame_n % self._chunksize, self._format))

        if self._format in self._cv2_fmts:
            if self._format == 'ppm':
                img = ensure_color(img)
            elif self._format == 'pgm':
                img = ensure_grayscale(img)
            cv2.imwrite(dest, img)
        elif self._format == 'npy':
            np.save(dest, img)
        elif self._format == 'bpk':
            bloscpack.pack_ndarray_file(img, dest)

        self._save_image_metadata(frame_number, frame_time)

    def _save_chunk(self, old, new):
        if old is not None:
            self._save_chunk_metadata(os.path.join(self._chunk_cdir, 'index'))
        if new is not None:
            self._chunk_cdir = os.path.join(self._basedir, '%06d' % new)
            os.mkdir(self._chunk_cdir)
            self._new_chunk_metadata(os.path.join(self._basedir, '%06d' % new, 'index'))

    def _find_chunks(self, chunk_numbers):
        if chunk_numbers is None:
            immediate_dirs = next(os.walk(self._basedir))[1]
            chunk_numbers = list(map(int, immediate_dirs))  # N.B. need list, as we iterate it twice
        return list(zip(chunk_numbers, tuple(os.path.join(self._basedir, '%06d' % int(n), 'index')
                                             for n in chunk_numbers)))

    # noinspection PyShadowingBuiltins
    @staticmethod
    def _open_image(path, format, color):
        if format in DirectoryImgStore._cv2_fmts:
            flags = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(path, flags)
        elif format == 'npy':
            img = np.load(path)
        elif format == 'bpk':
            with open(path, 'rb') as reader:
                img = bloscpack.numpy_io.unpack_ndarray(bloscpack.file_io.CompressedFPSource(reader))
        else:
            # Won't get here unless we relax checks in constructor, but better safe
            raise ValueError('unknown format %s' % format)
        return img

    def _load_image(self, idx):
        path = os.path.join(self._chunk_cdir, '%06d.%s' % (idx, self._format))
        img = self._open_image(path, self._format, self._color)
        return img, (self._chunk_md['frame_number'][idx], self._chunk_md['frame_time'][idx])

    def _load_chunk(self, n):
        cdir = os.path.join(self._basedir, '%06d' % n)
        if cdir != self._chunk_cdir:
            self._log.debug('loading chunk %s' % n)
            self._chunk_cdir = cdir
            self._load_chunk_metadata(os.path.join(self._chunk_cdir, 'index'))
            self._chunk_index = self._chunk_md['frame_number']

        self._chunk_n = n
        self._chunk_current_frame_idx = -1  # not used in DirectoryImgStore, but maintain compat

    @staticmethod
    def _extract_only_frame(basedir, chunk_n, frame_n, smd):
        fmt = smd['format']
        cdir = os.path.join(basedir, '%06d' % chunk_n)
        path = os.path.join(cdir, '%06d.%s' % (frame_n, fmt))

        imgshape = tuple(smd['imgshape'])
        color = (imgshape[-1] == 3) & (len(imgshape) == 3)

        return DirectoryImgStore._open_image(path, fmt, color)

    @classmethod
    def supported_formats(cls):
        fmts = list(cls._cv2_fmts) + list(cls._raw_fmts)
        if bloscpack is None:
            fmts.remove('bpk')
        return fmts

    @property
    def lossless(self):
        return self._format != 'jpg'


class VideoImgStore(_ImgStore):
    _supported_modes = 'wr'

    _cv2_fmts = {'mjpeg': FourCC('M', 'J', 'P', 'G'),
                 'mjpeg/avi': FourCC('M', 'J', 'P', 'G'),
                 'h264/mkv': FourCC('H', '2', '6', '4'),
                 'avc1/mp4': FourCC('a', 'v', 'c', '1')}

    def __init__(self, **kwargs):

        self._cap = None
        self._capfn = None

        fmt = kwargs.pop('format', None)
        # backwards compat
        if fmt == 'mjpeg':
            fmt = 'mjpeg/avi'

        # keep compat with VideoImgStoreFFMPEG
        seek = kwargs.pop('seek', None)

        if kwargs['mode'] == 'w':
            imgshape = kwargs['imgshape']

            if 'chunksize' not in kwargs:
                kwargs['chunksize'] = 500

            try:
                self._codec = self._cv2_fmts[fmt]
            except KeyError:
                raise ValueError('only %r supported', (self._cv2_fmts.keys(),))

            self._color = (imgshape[-1] == 3) & (len(imgshape) == 3)

            metadata = kwargs.get('metadata', {})
            metadata[STORE_MD_KEY] = {'format': fmt,
                                      'extension': '.%s' % fmt.split('/')[1]}
            kwargs['metadata'] = metadata
            kwargs['encoding'] = kwargs.pop('encoding', None)

        _ImgStore.__init__(self, **kwargs)

        self._supports_seeking = seek
        if self._supports_seeking:
            self._log.info('seeking is enabled on store')

        if self._mode == 'r':
            imgshape = self._metadata['imgshape']
            self._color = (imgshape[-1] == 3) & (len(imgshape) == 3)

    @staticmethod
    def _get_chunk_extension(metadata):
        # forward compatibility
        try:
            return metadata['extension']
        except KeyError:
            # backward compatibility with old mjpeg stores
            if metadata['format'] == 'mjpeg':
                return '.avi'
            # backwards compatibility with old bview/motif stores
            return '.mp4'

    @property
    def _ext(self):
        return self._get_chunk_extension(self._metadata)

    def _find_chunks(self, chunk_numbers):
        if chunk_numbers is None:
            avis = map(os.path.basename, glob.glob(os.path.join(self._basedir, '*%s' % self._ext)))
            chunk_numbers = list(map(int, map(operator.itemgetter(0), map(os.path.splitext, avis))))
        return list(zip(chunk_numbers, tuple(os.path.join(self._basedir, '%06d' % n) for n in chunk_numbers)))

    def _save_image(self, img, frame_number, frame_time):
        # we always write color because its more supported
        frame = ensure_color(img)
        self._cap.write(frame)
        if not os.path.isfile(self._capfn):
            raise Exception('The opencv backend does not actually have write support')
        self._save_image_metadata(frame_number, frame_time)

    def _save_chunk(self, old, new):
        if self._cap is not None:
            self._cap.release()
            self._save_chunk_metadata(os.path.join(self._basedir, '%06d' % old))

        if new is not None:
            fn = os.path.join(self._basedir, '%06d%s' % (new, self._ext))
            h, w = self._imgshape[:2]
            self._cap = cv2.VideoWriter(fn, self._codec, 25, (w, h), isColor=True)
            self._capfn = fn
            self._new_chunk_metadata(os.path.join(self._basedir, '%06d' % new))

    def _load_image(self, idx):
        if self._supports_seeking:
            # only seek if we have to, otherwise take the fast path
            if (idx - self._chunk_current_frame_idx) != 1:
                self._cap.set(getattr(cv2, "CAP_PROP_POS_FRAMES", 1), idx)
        else:
            if idx < self._chunk_current_frame_idx:
                self._load_chunk(self._chunk_n, _force=True)

            i = self._chunk_current_frame_idx + 1
            while i < idx:
                _, img = self._cap.read()
                i += 1

        _, _img = self._cap.read()
        if self._color:
            # almost certainly no-op as opencv usually returns color frames....
            img = ensure_color(_img)
        else:
            img = ensure_grayscale(_img)

        return img, (self._chunk_md['frame_number'][idx], self._chunk_md['frame_time'][idx])

    def _load_chunk(self, n, _force=False):
        fn = os.path.join(self._basedir, '%06d%s' % (n, self._ext))
        if _force or (fn != self._capfn):
            if self._cap is not None:
                self._cap.release()

            self._log.debug('loading chunk %s' % n)
            self._capfn = fn
            # noinspection PyArgumentList
            self._cap = cv2.VideoCapture(self._capfn)
            self._chunk_current_frame_idx = -1

            if not self._cap.isOpened():
                raise Exception("OpenCV unable to open %s" % fn)

            self._load_chunk_metadata(os.path.join(self._basedir, '%06d' % n))
            self._chunk_index = self._chunk_md['frame_number']

        self._chunk_n = n

    @classmethod
    def supported_formats(cls):
        # remove the duplicate
        fmts = list(cls._cv2_fmts.keys())
        fmts.remove('mjpeg')
        return fmts

    @classmethod
    def supports_format(cls, fmt):
        return fmt in cls._cv2_fmts

    @staticmethod
    def _extract_only_frame(basedir, chunk_n, frame_n, smd):
        capfn = os.path.join(basedir, '%06d%s' % (chunk_n,
                                                  VideoImgStore._get_chunk_extension(smd)))
        # noinspection PyArgumentList
        cap = cv2.VideoCapture(capfn)

        log = logging.getLogger('loopb.imgstore')
        log.debug('opending %s chunk %d frame_idx %d' % (capfn, chunk_n, frame_n))

        try:
            cap.set(getattr(cv2, "CAP_PROP_POS_FRAMES", 1), frame_n)
            _, img = cap.read()
            return img
        finally:
            cap.release()

    @property
    def lossless(self):
        return False

    def close(self):
        super(VideoImgStore, self).close()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._capfn = None

    def empty(self):
        _ImgStore.empty(self)
        for _, chunk_path in self._find_chunks(chunk_numbers=None):
            os.unlink(chunk_path + self._ext)
            self._remove_index(chunk_path)

    def insert_chunk(self, video_path, frame_numbers, frame_times, move=True):
        assert len(frame_numbers) == len(frame_times)
        assert video_path.endswith(self._ext)

        self._new_chunk_metadata(os.path.join(self._basedir, '%06d' % self._chunk_n))
        self._chunk_md['frame_number'] = np.asarray(frame_numbers)
        self._chunk_md['frame_time'] = np.asarray(frame_times)

        self._save_chunk_metadata(os.path.join(self._basedir, '%06d' % self._chunk_n))

        vid = os.path.join(self._basedir, '%06d%s' % (self._chunk_n, self._ext))
        if move:
            shutil.move(video_path, vid)
        else:
            shutil.copy(video_path, vid)

        self._chunk_n += 1


def new_for_filename(path, **kwargs):
    if path.endswith(STORE_MD_FILENAME) and (os.path.basename(path) == STORE_MD_FILENAME):
        basedir = os.path.dirname(path)
        fullpath = path
    elif os.path.isdir(path) and os.path.exists(os.path.join(path, STORE_MD_FILENAME)):
        basedir = path
        fullpath = os.path.join(path, STORE_MD_FILENAME)
    else:
        raise ValueError('should be a path to a store %s file or a directory containing one' % STORE_MD_FILENAME)

    if 'mode' not in kwargs:
        kwargs['mode'] = 'r'
    if 'basedir' not in kwargs:
        kwargs['basedir'] = basedir

    with open(fullpath, 'rt') as f:
        clsname = yaml.load(f)[STORE_MD_KEY]['class']

    # retain compatibility with internal loopbio stores
    if clsname == 'VideoImgStoreFFMPEG':
        clsname = 'VideoImgStore'

    try:
        cls = {DirectoryImgStore.__name__: DirectoryImgStore,
               VideoImgStore.__name__: VideoImgStore}[clsname]
    except KeyError:
        raise ValueError('store class %s not supported' % clsname)

    return cls(**kwargs)


def new_for_format(fmt, **kwargs):
    if 'mode' not in kwargs:
        kwargs['mode'] = 'w'
    for cls in (DirectoryImgStore, VideoImgStore):
        if cls.supports_format(fmt):
            kwargs['format'] = fmt
            return cls(**kwargs)
    raise ValueError('store class not found which supports format %s' % fmt)


def extract_only_frame(path, frame_index):
    smd = _extract_store_metadata(path)
    clsname = smd['class']

    if clsname == 'VideoImgStoreFFMPEG':
        clsname = 'VideoImgStore'

    try:
        cls = {DirectoryImgStore.__name__: DirectoryImgStore,
               VideoImgStore.__name__: VideoImgStore}[clsname]
    except KeyError:
        raise ValueError('store class %s not supported' % clsname)

    return cls.extract_only_frame(full_path=path, frame_index=frame_index, _smd=smd)


def get_supported_formats():
    f = []
    for cls in (DirectoryImgStore, VideoImgStore):
        f.extend(cls.supported_formats())
    return f
