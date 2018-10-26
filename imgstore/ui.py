from __future__ import print_function

import re
import sys
import subprocess

import cv2
import numpy as np

WINDOW_NORMAL = getattr(cv2, 'WINDOW_NORMAL', 0)
WINDOW_AUTOSIZE = getattr(cv2, 'WINDOW_AUTOSIZE', 1)

_SCREEN_RESOLUTION = None


def get_screen_resolution():
    """
    Returns: the screen resolution in pixels, as a string WxH
    """
    global _SCREEN_RESOLUTION

    if _SCREEN_RESOLUTION is None:
        resolution = ''

        try:
            if sys.platform == 'darwin':
                _re = re.compile(r"""^.*Resolution:\s*([\d\sx]+)""")
                out = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'])
                for l in out.splitlines():
                    _s = l.decode('utf-8')
                    m = _re.match(_s)
                    if m:
                        resolution = m.groups()[0]
            else:
                out = subprocess.check_output(['xrandr'])
                resolution_line = [l for l in out.splitlines() if '*' in l][0]
                resolution = resolution_line.split()[0]
        except subprocess.CalledProcessError as exc:
            print("ERROR detecting:", exc)
        except Exception as exc:
            print("ERROR parsing:", exc)

        _SCREEN_RESOLUTION = resolution

    return _SCREEN_RESOLUTION


def get_and_parse_screen_resolution(scale=1.0, default=(1024, 768)):
    res = get_screen_resolution()
    _w, _h = default

    if res:
        try:
            _w, _h = map(float, res.split('x'))
        except ValueError:
            print("ERROR splitting resolution string")
    else:
        print("ERROR unable to get resolution")

    w = float(scale) * _w
    h = float(scale) * _h
    return int(w), int(h)


def new_window(name, size=None, expanded_ui=True, shape=None, screen_relative_size=0.75):
    if shape is not None:
        size = shape[1], shape[0]

    flags = getattr(cv2, 'WINDOW_GUI_EXPANDED', 0) if expanded_ui else getattr(cv2, 'WINDOW_GUI_NORMAL', 16)
    if size is not None:
        cv2.namedWindow(name, WINDOW_NORMAL | flags)

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
        cv2.namedWindow(name, WINDOW_AUTOSIZE | flags)
        sizestr = 'auto'

    print("CREATED WINDOW %s:%s (from %r)" % (sizestr, flags, size))
    return name, sizestr
