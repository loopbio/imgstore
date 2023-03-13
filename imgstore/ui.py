from __future__ import print_function

import re
import sys
import logging
import subprocess

import cv2
import numpy as np

_SCREEN_RESOLUTION = None
_IS_MAC = sys.platform == 'darwin'

_log = logging.getLogger('imgstore.ui')


class _Window(object):
    def __init__(self, name, flags, sizestr):
        self.name = name
        self._flags = flags
        self._size = sizestr
        self._set = False

    def __repr__(self):
        return "Window<'%s', flags=%s, sizestr=%s, mac=%s>" % (self.name, bin(self._flags),
                                                               self._size, _IS_MAC)

    def __str__(self):
        return self.name

    @property
    def size(self):
        if 'x' in self._size:
            return tuple(map(int, self._size.split('x')))
        return None

    def imshow(self, *args, **kwargs):
        cv2.imshow(*args, **kwargs)
        # Mac requires window size to be set after the first imshow
        if _IS_MAC and not self._set:
            sz = self.size
            if sz is not None:
                w, h = sz
                cv2.resizeWindow(self.name, int(w), int(h))
                self._set = True

    # noinspection PyPep8Naming
    def waitKey(self, *args, **kwargs):
        return cv2.waitKey(*args, **kwargs)


def get_screen_resolution():
    """
    Returns: the screen resolution in pixels, as a string WxH
    """
    global _SCREEN_RESOLUTION

    if _SCREEN_RESOLUTION is None:
        resolution = ''

        try:
            if _IS_MAC:
                _re = re.compile(r"""^.*Resolution:\s*([\d\sx]+)""")
                out = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'])
                for l in out.splitlines():
                    _s = l.decode('utf-8')
                    m = _re.match(_s)
                    if m:
                        resolution = m.groups()[0]
            else:
                out = subprocess.check_output(['xrandr'], text=True)
                resolution_line = [l for l in out.splitlines() if '*' in l][0]
                resolution = resolution_line.split()[0]
        except subprocess.CalledProcessError as exc:
            _log.warn("could not detect resolution: %r:" % exc)
        except Exception as exc:
            _log.warn("could not parse resolution: %r:" % exc)

        _SCREEN_RESOLUTION = resolution

    return _SCREEN_RESOLUTION


def get_and_parse_screen_resolution(scale=1.0, default=(1024, 768)):
    res = get_screen_resolution()
    _w, _h = default

    if res:
        try:
            _w, _h = map(float, res.split('x'))
        except ValueError:
            _log.warn("could not splitting resolution string: %r:" % res)
    else:
        _log.warn("could not get resolution string")

    w = float(scale) * _w
    h = float(scale) * _h
    return int(w), int(h)


def new_window(name, size=None, expanded_ui=True, shape=None, screen_relative_size=0.75):
    if shape is not None:
        size = shape[1], shape[0]

    flags = cv2.WINDOW_GUI_EXPANDED if expanded_ui else cv2.WINDOW_GUI_NORMAL
    if size is not None:
        flags = cv2.WINDOW_NORMAL | flags
        cv2.namedWindow(name, flags)

        if (size[0] > 0) and (not np.isnan(size[0])):

            # create a resizable window but limit its size to 75% the screen size
            sw, sh = get_and_parse_screen_resolution(scale=screen_relative_size, default=(1024, 768))

            if np.isinf(size[0]):
                w = sw
                h = sh
            else:
                w, h = size

                sfw = (float(sw) / w) if w > sw else 1.0
                sfh = (float(sh) / h) if h > sh else 1.0

                sf = min(sfw, sfh)

                h *= sf
                w *= sf

            cv2.resizeWindow(name, int(w), int(h))
            sizestr = '%dx%d' % (int(w), int(h))
        else:
            sizestr = 'unknown'
    else:
        flags = cv2.WINDOW_AUTOSIZE | flags
        cv2.namedWindow(name, flags)
        sizestr = 'auto'

    win = _Window(name, flags, sizestr)
    return win
