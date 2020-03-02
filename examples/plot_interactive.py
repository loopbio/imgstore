from __future__ import print_function

import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from imgstore import new_for_filename

figimg = plt.figure()
aximg = figimg.add_subplot(111)
aximg.set_axis_off()

try:
    SOURCE = sys.argv[1]
except IndexError:
    SOURCE = '/mnt/loopbio/tests/motif/mcc8_20191231_155718/'

store = new_for_filename(SOURCE)
assert store.has_extra_data

df = store.get_extra_data(ignore_corrupt_chunks=True)

fts = np.asarray(df['frame_time'])
fns = np.asarray(df['frame_number'])


print("store video frame range: %s -> %s" % (store.frame_min, store.frame_max))
print("extra data frame rage: %s -> %s" % (fns.min(), fns.max()))

USE_CAMERA_TIME = True
sts = np.asarray(df['sample_time'])
sample_delay = np.asarray(df['sample_delay'])

print("mean delay", sample_delay.mean())


def imshowu8(_img, _ax):
    _is_color = (_img.shape[-1] == 3) & (_img.ndim == 3)
    if _is_color:
        _ax.imshow(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB), aspect='equal')
    else:
        _ax.imshow(_img, aspect='equal', vmin=0, vmax=255, cmap='gray')


def on_plotclick(event):

    if event.button == 1:
        if (event.xdata is not None) and (event.xdata > 0):
            idx = int(event.xdata)

            fn_orig = fns[idx]

            delay = sample_delay[idx]
            if USE_CAMERA_TIME:
                ts = fts[idx]
            else:
                ts = sts[idx]

            img, (fn, ft) = store.get_nearest_image(ts - delay)

            print('fn (delay corrected): %s vs %s' % (fn, fn_orig))

            figimg.suptitle("frame_number:%s\nframe_time:%s" % (fn, ft))
            imshowu8(img, aximg)
            figimg.canvas.draw()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(store.get_frame_metadata()['frame_number'])

figdata = plt.figure(figsize=(8, 8))
figdata.canvas.mpl_connect('button_press_event', on_plotclick)

ax = figdata.add_subplot(111)
# todo: axvline the store range?

df.plot(subplots=True, ax=ax, sharex=True, style=',' if len(df) > 1e6 else '-')

plt.show()
