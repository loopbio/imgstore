from __future__ import print_function, absolute_import

import cv2

from .stores import new_for_filename, get_supported_formats, new_for_format, STORE_MD_FILENAME
from .ui import new_window


def main_viewer():
    import os.path
    import argparse
    import logging

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

    win = new_window('imgstore', shape=store.image_shape)
    print("Created %r" % win)

    while True:
        try:
            img, _ = store.get_next_image()
        except EOFError:
            break

        win.imshow('imgstore', img)

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

