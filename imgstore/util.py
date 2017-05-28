import re
import datetime
import collections

import cv2
import json

import numpy as np


if cv2.__version__.startswith('3.'):
    FourCC = cv2.VideoWriter_fourcc
else:
    # noinspection PyUnresolvedReferences
    FourCC = cv2.cv.CV_FOURCC


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
        # pylon disagrees with opencv
        if s in {'BayerBG', 'Bayer_BG'}:
            code = 'cv_bayerrg'
        elif s in {'BayerRG', 'Bayer_RG'}:
            code = 'cv_bayerbg'
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


def main_viewer():
    import os.path
    import argparse
    import logging

    from imgstore.stores import new_for_filename, STORE_MD_FILENAME

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1)
    parser.add_argument('--fps', default=30., type=float,
                        help='playback store at this speed (0 = play back as fast as possible)')
    args = parser.parse_args()

    if args.fps == 0.0:
        args.fps = 1000.0  # ensure w sleep at least 1ms

    path = args.path[0]
    if os.path.isdir(path):
        path = os.path.join(path, STORE_MD_FILENAME)
    if not os.path.isfile(path):
        parser.error('path must be an imagestore directory')

    store = new_for_filename(path)

    while True:
        try:
            img, _ = store.get_next_image()
        except EOFError:
            break

        cv2.imshow('imgstore', img)

        k = cv2.waitKey(int(1000. / args.fps)) & 0xFF
        if k == 27:
            break


def main_saver():
    import os.path
    import argparse
    import logging
    import time
    import itertools
    import errno

    from imgstore.stores import get_supported_formats, new_for_format

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('dest', nargs=1)
    parser.add_argument('--format',
                        choices=get_supported_formats(), default='mjpeg')
    args = parser.parse_args()

    # noinspection PyArgumentList
    cap = cv2.VideoCapture(0)
    _, img = cap.read()

    path = args.dest[0]
    if os.path.exists(path):
        parser.error('destination already exists')
    else:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                parser.error('could not create destination path')

    store = new_for_format(args.format,
                           basedir=path, mode='w',
                           imgdtype=img.dtype,
                           imgshape=img.shape)

    for i in itertools.count():
        _, img = cap.read()

        cv2.imshow('preview', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        store.add_image(img, i, time.time())

    store.close()
