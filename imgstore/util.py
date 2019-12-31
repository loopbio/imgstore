import re
import datetime
import collections

import cv2
import json

import numpy as np


if cv2.__version__.startswith(('3.', '4.')):
    FourCC = cv2.VideoWriter_fourcc
else:
    # noinspection PyUnresolvedReferences
    FourCC = cv2.cv.CV_FOURCC


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


def ensure_grayscale(img, ensure_copy=False):
    return _cvt_color(img, cv2.COLOR_BGR2GRAY, ensure_copy=ensure_copy)


def ensure_color(img, ensure_copy=False):
    return _cvt_color(img, cv2.COLOR_GRAY2BGR, ensure_copy=ensure_copy)


class ImageCodecProcessor(object):
    """
    Allows converting a number of image encodings (Bayer, YUV, etc)
    to OpenCV native format (BGR).

    Because I think OpenCV disagrees about what to name bayer formats
    as compared to other libraries, all encodings are prefixed with
    'cv_'
    """

    # https://regex101.com/r/pL2mY2/1
    CV2_CODEC_RE = re.compile("^COLOR_(?P<codec>[A-Za-z0-9_]+)2BGR($|(?P<method>_[A-Za-z0-9]*))")

    def __init__(self):
        self._default_code = None
        self._default_enum = None

        self._cv2_code_to_enum = {}
        self._cv2_enum_to_code = {}
        # an extra private mapping because OpenCV has many enums mapping to the same value
        self.__cv2_enum_to_name = collections.defaultdict(list)
        for name in dir(cv2):
            m = self.CV2_CODEC_RE.match(name)
            if m:
                enum = int(getattr(cv2, name))
                self.__cv2_enum_to_name[enum].append(name)

                parts = m.groupdict()
                code = 'cv_%s' % parts['codec'].lower().replace('_', '')

                meth = parts.get('method', None)
                if meth:
                    if 'bayer' in code:
                        # cv includes different bayer methods, ignore them
                        continue
                    elif meth == 'full':
                        # fixme: not sure what this is
                        continue

                    code += '_%s' % parts['method'].lower().replace('_', '')

                self._cv2_code_to_enum[code] = enum
                self._cv2_enum_to_code[enum] = code

        self.autoconvert = lambda x: x

        # pprint.pprint(self._cv2_code_to_enum)
        # pprint.pprint(self._cv2_enum_to_code)

    @classmethod
    def from_pylon_format(cls, s):
        # pylon disagrees with opencv (B & R are always switched)
        if s in {'BayerBG', 'Bayer_BG'}:
            code = 'cv_bayerrg'
        elif s in {'BayerRG', 'Bayer_RG'}:
            code = 'cv_bayerbg'
        elif s in {'BayerGB', 'Bayer_GB'}:
            code = 'cv_bayergr'
        elif (not s) or (s == 'None'):
            code = None
        else:
            raise ValueError('unknown pylon format: %r' % s)

        obj = cls()
        obj.set_default_code(code)
        return obj

    @classmethod
    def from_cv2_enum(cls, e):
        obj = cls()
        obj.set_default_code(obj.cv2_enum_to_code(e))
        return obj

    @property
    def encoding(self):
        return self._default_code or ''

    def _autodecode_cvtcolor(self, img):
        return cv2.cvtColor(img, self._default_enum)

    def set_default_code(self, code):
        if not code:
            self._default_code = None
            self._default_enum = None
            self.autoconvert = lambda x: x
        else:
            if not self.check_code(code):
                raise ValueError('unknown image encoding:' % code)
            self._default_code = code
            self._default_enum = self._cv2_code_to_enum[code]
            self.autoconvert = self._autodecode_cvtcolor

    def check_code(self, code):
        return code in self._cv2_code_to_enum

    def cv2_enum_to_code(self, enum):
        return self._cv2_enum_to_code[enum]

    def convert_to_bgr(self, img, code=None, method_enum=None):
        if code is None:
            code = self._default_code
        if method_enum is None:
            method_enum = self._cv2_code_to_enum[code]
        if code is None:
            raise ValueError('encoding must be specified')
        return cv2.cvtColor(img, method_enum)


class JsonCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # noinspection PyUnresolvedReferences
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (complex, np.complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()

        return json.JSONEncoder.default(self, obj)


def motif_extra_data_h5_to_df(path):
    import h5py
    import pandas as pd

    with h5py.File(path, 'r') as f:
        dat = {}

        # the camera information is stored in an array with compound datatype
        camera = np.asarray(f['camera'])

        # recording can be stopped before chunk is full, so trim away rows in the store that
        # were pre-allocated, but not recorded. pre-allocated but un-used frames are indicated with
        # a framenumber < 0
        mask = camera['frame_number'] >= 0

        # motif stores the names of datasets in a root attribute
        datasets = map(str.strip, f.attrs['datasets'].split(','))

        for dsname in datasets:
            ds = f[dsname]
            col_names = map(str.strip, ds.attrs['column_names'].split(','))
            # trim the array
            arr = ds[..., mask]

            assert len(col_names) == arr.shape[0]

            for col, col_name in enumerate(col_names):
                dat[col_name] = arr[col]

            # also add the camera sync information
            for cam_md_name in camera.dtype.names:
                dat[cam_md_name] = camera[cam_md_name][mask]

        return pd.DataFrame(dat)
