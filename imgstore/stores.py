# coding=utf-8
from __future__ import print_function, division, absolute_import
import os.path
import itertools
import operator
import time
import logging
import glob
import uuid
import string

import cv2
from ruamel import yaml
import json
import numpy as np
import pandas as pd

try:
    import bloscpack
except ImportError:
    bloscpack = None

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'r+b')

from .util import ImageCodecProcessor, JsonCustomEncoder, FourCC


def _cvt_color(img, code, ensure_copy=True):
    # protect against empty last dimensions
    _is_color = (img.shape[-1] == 3) & (img.ndim == 3)

    if code == cv2.COLOR_GRAY2BGR:
        if _is_color:
            return img.copy() if ensure_copy else img
        else:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif code == cv2.COLOR_BGR2GRAY:
        if _is_color:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            return img.copy() if ensure_copy else img
    else:
        return ValueError("cvtColor code not understood: %s" % code)


def _ensure_grayscale(img):
    return _cvt_color(img, cv2.COLOR_BGR2GRAY, ensure_copy=False)


def _ensure_color(img):
    return _cvt_color(img, cv2.COLOR_GRAY2BGR, ensure_copy=False)


STORE_MD_KEY = '__store'
STORE_MD_FILENAME = 'metadata.yaml'


class _ImgStore(object):

    _version = 2
    _supported_modes = ''

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
            # the chunk index is a list of framenumbers, the index of the
            # framenumber is the position in the chunk
            self._chunk_index = []
            self._init_read()

            t0 = time.time()
            self._chunk_n_and_chunk_paths = self._find_chunks(chunk_numbers=None)
            self._log.debug('found %s chunks in in %fs' % (len(self._chunk_n_and_chunk_paths), time.time() - t0))

            self._build_index(self._chunk_n_and_chunk_paths)

            # reset to the start of the file and load the first chunk
            self._frame_idx = -1
            self._load_chunk(0)
            self.frame_number = self.frame_min

    def _init_read(self):
        with open(os.path.join(self._basedir, STORE_MD_FILENAME), 'rt') as f:
            allmd = yaml.load(f, Loader=yaml.Loader)
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
            uuid = smd['uuid']
        except KeyError:
            self._log.warn('source is missing uuid, generating a weak one from filename')
            uuid = string.ljust(os.path.basename(self._basedir), 32, 'X')
        self._uuid = uuid

        self._imgshape = tuple(smd['imgshape'])
        self._imgdtype = smd['imgdtype']
        self._chunksize = int(smd['chunksize'])
        self._encoding = smd['encoding']
        self._metadata = smd

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

        store_md = {'imgshape': imgshape,
                    'imgdtype': self._imgdtype,
                    'chunksize': chunksize,
                    'class': self.__class__.__name__,
                    'version': self._version,
                    'encoding': encoding,
                    'uuid': uuid.uuid4().hex}

        if metadata is None:
            metadata[STORE_MD_KEY] = store_md
        elif isinstance(metadata, dict):
            try:
                metadata[STORE_MD_KEY].update(store_md)
            except KeyError:
                metadata[STORE_MD_KEY] = store_md
        else:
            raise ValueError('metadata must be a dictionary')

        with open(os.path.join(self._basedir, STORE_MD_FILENAME), 'wt') as f:
            yaml.safe_dump(metadata, f)

        # noinspection PyUnresolvedReferences
        smd = metadata.pop(STORE_MD_KEY)
        self._metadata = smd
        self._user_metadata.update(metadata)

        self._save_chunk(None, self._chunk_n)

    def _save_image(self, img, frame_number, frame_time):
        raise NotImplementedError

    def _save_chunk(self, old, new):
        raise NotImplementedError

    def _load_image(self, idx):
        raise NotImplementedError

    def _load_chunk(self, n):
        raise NotImplementedError

    def _build_index(self, chunk_n_and_chunk_paths):
        raise NotImplementedError

    def _find_chunks(self, chunk_numbers):
        raise NotImplementedError

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
    def image_shape(self):
        # if encoding is specified, we always decode to bgr (color)
        if self._encoding:
            return self._imgshape[0], self._imgshape[1], 3
        else:
            return self._imgshape

    @property
    def duration(self):
        return self._tN - self._t0

    @property
    def full_path(self):
        return os.path.join(self._basedir, STORE_MD_FILENAME)

    def disable_decoding(self):
        self._decode_image = lambda x: x

    def add_extra_data(self, **data):
        pass

    # noinspection PyMethodMayBeStatic
    def get_extra_data(self):
        return {}

    def add_image(self, img, frame_number, frame_time):
        self._save_image(self._encode_image(img), frame_number, frame_time)

        self.frame_max = np.nanmax((frame_number, self.frame_max))
        self.frame_min = np.nanmin((frame_number, self.frame_min))
        self.frame_number = frame_number
        self.frame_time = frame_time

        if self._frame_n == 0:
            self._t0 = time.time()
        self._tN = time.time()

        self._frame_n += 1
        if (self._frame_n % self._chunksize) == 0:
            old = self._chunk_n
            new = self._chunk_n + 1
            self._save_chunk(old, new)
            self._chunk_n = new

        self.frame_count = self._frame_n

    def _get_next_framenumber_and_idx(self):
        if self.frame_number == self.frame_max:
            raise EOFError

        idx = self._frame_idx + 1
        try:
            frame_number = self._chunk_index[idx]
        except IndexError:
            # open the next chunk
            chunk_n = self._chunk_n + 1
            self._frame_idx = -1
            self._load_chunk(chunk_n)
            self._chunk_n = chunk_n
            # first frame is start of chunk
            idx = 0
            frame_number = self._chunk_index[idx]

        return frame_number, idx

    def get_next_framenumber(self):
        return self._get_next_framenumber_and_idx()[0]

    def get_next_image(self):
        frame_number, idx = self._get_next_framenumber_and_idx()
        return self.get_image(frame_number, exact_only=True, frame_idx=idx)

    def get_image(self, frame_number, exact_only=True, frame_idx=None):
        # there is a high likelihood the current chunk holds the next frame
        # so look there first
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

            self._frame_idx = -1
            self._load_chunk(chunk_n)
            self._chunk_n = chunk_n
            try:
                frame_idx = self._chunk_index.index(frame_number)
            except ValueError:
                raise ValueError('%s %s not found in chunk %s' % ('frame_number', frame_number, chunk_n))

        # ensure the read works before setting frame_number
        _img, (_frame_number, _frame_timestamp) = self._load_image(frame_idx)
        img = self._decode_image(_img)

        self._frame_idx = frame_idx
        self.frame_number = frame_number

        return img, (_frame_number, _frame_timestamp)

    def close(self):
        if self._mode in 'wa':
            self._save_chunk(self._chunk_n, None)

    # noinspection PyMethodMayBeStatic
    def get_frame_metadata(self):
        return {}

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


# noinspection PyUnresolvedReferences,PyAttributeOutsideInit,PyClassHasNoInit
class _MetadataMixin:

    FRAME_MD = ('frame_number', 'frame_time')

    def _save_index(self, path_with_extension, data_dict):
        _, extension = os.path.splitext(path_with_extension)
        if extension == '.yaml':
            with open(path_with_extension, 'w') as f:
                yaml.safe_dump(data_dict, f)
        elif extension == '.npz':
            with open(path_with_extension, 'wb') as f:
                # noinspection PyTypeChecker
                np.savez(f, **data_dict)
        else:
            raise ValueError('unknown index format: %s' % extension)

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
        self._chunk_md = {k: [] for k in _MetadataMixin.FRAME_MD}
        self._chunk_md.update(self._metadata)

    def _save_image_metadata(self, frame_number, frame_time):
        self._chunk_md['frame_number'].append(frame_number)
        self._chunk_md['frame_time'].append(frame_time)

    def add_extra_data(self, **data):
        if not data:
            return

        data['frame_time'] = self.frame_time
        data['frame_number'] = self.frame_number

        # noinspection PyBroadException
        try:
            txt = json.dumps(data, cls=JsonCustomEncoder)
        except:
            self._log.warning('error writing extra data', exc_info=True)
            return

        if self._extra_data_fp is None:
            self._extra_data_fp = open(self._extra_data_fn, 'wt')
            self._extra_data_fp.write('[')
        else:
            self._extra_data_fp.write(', ')
        self._extra_data_fp.write(txt)

    # noinspection PyMethodMayBeStatic
    def _remove_index(self, path_without_extension):
        for extension in ('.npz', '.yaml'):
            path = path_without_extension + extension
            if os.path.exists(path):
                os.unlink(path)

    # noinspection PyMethodMayBeStatic
    def _load_index(self, path_without_extension):
        for extension in ('.npz', '.yaml'):
            path = path_without_extension + extension
            if os.path.exists(path):
                if extension == '.yaml':
                    with open(path, 'rt') as f:
                        dat = yaml.safe_load(f)
                        return {k: dat[k] for k in _MetadataMixin.FRAME_MD}
                elif extension == '.npz':
                    dat = np.load(path)
                    return {k: dat[k].tolist() for k in _MetadataMixin.FRAME_MD}

        raise IOError('could not find index %s' % path_without_extension)

    def _build_index(self, chunk_n_and_chunk_paths):
        t0 = time.time()
        for chunk_n, chunk_path in chunk_n_and_chunk_paths:

            try:
                idx = self._load_index(chunk_path)
            except IOError:
                self._log.warning('missing index for chunk %s' % chunk_n)
                continue

            if not idx['frame_number']:
                # empty chunk
                continue

            self.frame_count += len(idx['frame_number'])

            for frame_range in _extract_ranges(idx['frame_number']):
                self._index[frame_range] = chunk_n

        self._log.debug('built index in %fs' % (time.time() - t0))

        self.frame_min = np.nan if 0 == len(self._index) else min(low for low, _ in self._index)
        self.frame_max = np.nan if 0 == len(self._index) else max(high for _, high in self._index)

        self._log.debug('frame range %f -> %f' % (self.frame_min, self.frame_max))

    def _load_chunk_metadata(self, path_without_extension):
        self._chunk_md = self._load_index(path_without_extension)

    def get_frame_metadata(self):
        dat = {k: [] for k in _MetadataMixin.FRAME_MD}
        for chunk_n, chunk_path in self._find_chunks(self.chunks):
            idx = self._load_index(chunk_path)
            for k in _MetadataMixin.FRAME_MD:
                dat[k].extend(idx[k])
        return dat

    def get_extra_data(self):
        dfs = []
        for chunk_n, chunk_path in self._chunk_n_and_chunk_paths:
            path = chunk_path + '.extra.json'
            if os.path.exists(path):
                dfs.append(pd.read_json(path, orient='record'))
        return pd.concat(dfs, axis=0, ignore_index=True)


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


class DirectoryImgStore(_MetadataMixin, _ImgStore):

    _supported_modes = 'wr'

    _cv2_fmts = {'tif', 'png', 'jpg', 'ppm', 'pgm', 'bmp'}
    _raw_fmts = {'npy', 'bpk'}

    def __init__(self, **kwargs):

        self._chunk_cdir = ''
        self._chunk_md = {}

        # keep compat with VideoImgStoreFFMPEG
        kwargs.pop('seek', None)

        if kwargs['mode'] == 'w':
            if 'chunksize' not in kwargs:
                kwargs['chunksize'] = 100
            self._format = kwargs.pop('format')
            metadata = kwargs.get('metadata', {})
            metadata[STORE_MD_KEY] = {'format': self._format}
            kwargs['metadata'] = metadata
            kwargs['encoding'] = kwargs.pop('encoding', None)

        _ImgStore.__init__(self, **kwargs)

        if self._mode == 'r':
            self._format = self._metadata['format']

        self._color = (self._imgshape[-1] == 3) & (len(self._imgshape) == 3)

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
                img = _ensure_color(img)
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

    def _load_image(self, idx):
        path = os.path.join(self._chunk_cdir, '%06d.%s' % (idx, self._format))
        if self._format in self._cv2_fmts:
            flags = cv2.IMREAD_COLOR if self._color else cv2.IMREAD_GRAYSCALE
            img = cv2.imread(path, flags)
        elif self._format == 'npy':
            img = np.load(path)
        elif self._format == 'bpk':
            with open(path, 'rb') as reader:
                img = bloscpack.numpy_io.unpack_ndarray(bloscpack.file_io.CompressedFPSource(reader))
        else:
            # Won't get here unless we relax checks in constructor, but better safe
            raise ValueError('unknown format %s' % self._format)
        return img, (self._chunk_md['frame_number'][idx], self._chunk_md['frame_time'][idx])

    def _load_chunk(self, n):
        cdir = os.path.join(self._basedir, '%06d' % n)
        if cdir != self._chunk_cdir:
            self._log.debug('loading chunk %s' % n)
            self._chunk_cdir = cdir
            self._load_chunk_metadata(os.path.join(self._chunk_cdir, 'index'))
            self._chunk_index = self._chunk_md['frame_number']

    @classmethod
    def supported_formats(cls):
        return list(cls._cv2_fmts) + list(cls._raw_fmts)

    @classmethod
    def supports_format(cls, fmt):
        return (fmt in cls._cv2_fmts) or (fmt in cls._raw_fmts)

    @property
    def lossless(self):
        return self._format != 'jpg'


class VideoImgStore(_MetadataMixin, _ImgStore):

    _supported_modes = 'wr'

    _cv2_fmts = {'mjpeg': FourCC('M', 'J', 'P', 'G'),
                 'mjpeg/avi': FourCC('M', 'J', 'P', 'G'),
                 'h264/mkv': FourCC('H', '2', '6', '4')}

    def __init__(self, **kwargs):

        self._cap = None
        self._capfn = None

        fmt = kwargs.pop('format', None)
        # backwards compat
        if fmt == 'mjpeg':
            fmt = 'mjpeg/avi'

        # keep compat with VideoImgStoreFFMPEG
        kwargs.pop('seek', None)

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

        if self._mode == 'r':
            imgshape = self._metadata['imgshape']
            self._color = (imgshape[-1] == 3) & (len(imgshape) == 3)

    @property
    def lossless(self):
        return False

    @property
    def _ext(self):
        # forward compatibility
        try:
            return self._metadata['extension']
        except KeyError:
            # backward compatibility with old mjpeg stores
            if self._metadata['format'] == 'mjpeg':
                return '.avi'
            # backwards compatibility with old bview/motif stores
            return '.mp4'

    def _find_chunks(self, chunk_numbers):
        if chunk_numbers is None:
            avis = map(os.path.basename, glob.glob(os.path.join(self._basedir, '*%s' % self._ext)))
            chunk_numbers = list(map(int, map(operator.itemgetter(0), map(os.path.splitext, avis))))
        return list(zip(chunk_numbers, tuple(os.path.join(self._basedir, '%06d' % n) for n in chunk_numbers)))

    def _save_image(self, img, frame_number, frame_time):
        # we always write color because its more supported
        frame = _ensure_color(img)
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
        # only seek if we have to, otherwise take the fast path
        if (idx - self._frame_idx) != 1:
            self._cap.set(getattr(cv2, "CAP_PROP_POS_FRAMES", 1), idx)

        _, _img = self._cap.read()
        if self._color:
            # almost certainly no-op as opencv usually returns color frames....
            img = _ensure_color(_img)
        else:
            img = _ensure_grayscale(_img)

        return img, (self._chunk_md['frame_number'][idx], self._chunk_md['frame_time'][idx])

    def _load_chunk(self, n):
        fn = os.path.join(self._basedir, '%06d%s' % (n, self._ext))
        if fn != self._capfn:
            if self._cap is not None:
                self._cap.release()

            self._log.debug('loading chunk %s' % n)
            self._capfn = fn
            # noinspection PyArgumentList
            self._cap = cv2.VideoCapture(self._capfn)

            if not self._cap.isOpened():
                raise Exception("OpenCV unable to open %s" % fn)

            self._load_chunk_metadata(os.path.join(self._basedir, '%06d' % n))
            self._chunk_index = self._chunk_md['frame_number']

    @classmethod
    def supported_formats(cls):
        return cls._cv2_fmts.keys()

    @classmethod
    def supports_format(cls, fmt):
        return fmt in cls._cv2_fmts


def new_for_filename(path, **kwargs):
    filename = os.path.basename(path)
    if filename != STORE_MD_FILENAME:
        raise ValueError('should be a path to a store %s file' % STORE_MD_FILENAME)

    if 'mode' not in kwargs:
        kwargs['mode'] = 'r'
    if 'basedir' not in kwargs:
        kwargs['basedir'] = os.path.dirname(path)

    with open(path, 'rt') as f:
        clsname = yaml.load(f, Loader=yaml.Loader)[STORE_MD_KEY]['class']

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
    for cls in (DirectoryImgStore, VideoImgStore):
        if cls.supports_format(fmt):
            kwargs['format'] = fmt
            return cls(**kwargs)
    raise ValueError('store class not found which supports format %s' % fmt)


def get_supported_formats():
    f = []
    for cls in (DirectoryImgStore, VideoImgStore):
        f.extend(cls.supported_formats())
    return f
