from __future__ import print_function, absolute_import
import logging

import cv2
import numpy as np

from .stores import new_for_filename, get_supported_formats, new_for_format
from .ui import new_window
from .util import motif_get_parse_true_fps, motif_get_recording_timezone
from .font import FontRendererStatusBar

_log = logging.getLogger('imgstore.apps')


def main_viewer():
    import argparse
    import logging
    import datetime

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1)
    parser.add_argument('--fps', default=30., type=float,
                        help='playback store at this speed (0 = play back as fast as possible)')
    parser.add_argument('--hide-statusbar', default=False, action='store_true',
                        help='hide status bar which shows frame time/number')
    args = parser.parse_args()

    if args.fps == 0.0:
        args.fps = 1000.0  # ensure w sleep at least 1ms

    store = new_for_filename(args.path[0])
    tz = motif_get_recording_timezone(store)

    def _draw_statusbar_nop(_img, _md):
        pass

    draw_statusbar = _draw_statusbar_nop

    if not args.hide_statusbar:
        _rend = None
        try:
            _ltxt_fmt = '{:%d}' % (len('%d' % int(store.frame_max)))
            def _get_status_bar_txt(_frame_number, _frame_time):
                return _ltxt_fmt.format(_frame_number), datetime.datetime.fromtimestamp(_frame_time, tz=tz).isoformat()

            _ltxt, _rtxt = _get_status_bar_txt(store.frame_max, store.frame_time_max)
            _rend = FontRendererStatusBar.new_for_image_size((store.image_shape[1], store.image_shape[0]),
                                                             ltext=_ltxt, rtext=_rtxt)
        except Exception:
            _log.error('could not construct statusbar', exc_info=True)

        def _draw_statusbar_actual(_img, _md):
            if _rend is not None:
                _ltxt, _rtxt = _get_status_bar_txt(*md)
                return _rend.render(img=_img, ltext=_ltxt, rtext=_rtxt)
            return _img

        draw_statusbar = _draw_statusbar_actual

    _log.info('true fps: %s' % motif_get_parse_true_fps(store))
    _log.info('fr fps: %s' % (1.0 / np.median(np.diff(store._get_chunk_metadata(0)['frame_time']))))

    win = new_window('imgstore', shape=store.image_shape)
    _log.debug('created window: %r' % win)

    paused = False
    while True:
        if not paused:
            try:
                img, md = store.get_next_image()
            except EOFError:
                break

        try:
            draw_statusbar(img, md)
        except:
            pass

        win.imshow('imgstore', img)

        k = cv2.waitKey(int(1000. / args.fps)) & 0xFF
        if (k == 27) or (k == ord('q')):
            break

        if k == ord(' '):
            paused ^= 1


def main_saver():
    import os.path
    import argparse
    import logging
    import time
    import itertools
    import errno

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('dest', nargs=1)
    parser.add_argument('--source', type=str, default=0,
                        metavar='PATH',
                        help='path to a file to convert to an imgstore')
    parser.add_argument('--format',
                        choices=get_supported_formats(), default='mjpeg')
    args = parser.parse_args()

    # noinspection PyArgumentList
    cap = cv2.VideoCapture(args.source)
    _, img = cap.read()

    if img is None:
        parser.error('could not open source: %s' % args.source)

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


def main_test():
    import pytest
    import os.path

    pytest.main(['-v', os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tests')])


def generate_timecodes(store, dest_file):
    _ts = np.asarray(store.get_frame_metadata()['frame_time'])
    ts = _ts - _ts[0]

    dest_file.write('# timecode format v2\n')
    for t in ts:
        dest_file.write('{0:.3f}\n'.format(t * 1000.))
    return ts


def main_generate_timecodes():
    import sys
    import argparse
    import logging

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1)
    args = parser.parse_args()

    store = new_for_filename(args.path[0])

    generate_timecodes(store, sys.stdout)
