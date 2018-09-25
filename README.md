IMGStore - Houses Your Video And Data
=====================================

Imgstore is a container for video frames and metadata. It allows efficient storage and seeking
through recordings from hours to weeks in duration. It supports compressed and uncompressed formats.

Imgstore allows reading (and writing) videos recorded with
loopbio's [Motif](http://loopbio.com/recording/) recording system.

# The Concept

Video data is broken into chunks, which can be individual video files `VideoImgStore`, or
a directory full of images `DirectoryImgStore`. The format of the chunks determines if the store is
compressed, uncompressed, lossless or lossy.

## Example: Write a store

```python
import imgstore
import numpy as np
import cv2
import time

height = width = 500
blank_image = np.zeros((height,width,3), np.uint8)

store = imgstore.new_for_format('jpg', mode='w', basedir='mystore',
                                imgshape=blank_image.shape, imgdtype=blank_image.dtype,
                                chunksize=1000)

for i in range(40):
    img = blank_image.copy()
    cv2.putText(img,str(i),(0,300), cv2.FONT_HERSHEY_SIMPLEX, 4, 255)
    store.add_image(img, i, time.time())

store.close()
```

## Example: Read a store

```python
from imgstore import new_for_filename

store = new_for_filename('mystore/metadata.yaml')

print 'frames in store:', store.frame_count
print 'min frame number:', store.frame_min
print 'max frame number:', store.frame_max

# read first frame
img, (frame_number, frame_timestamp) = store.get_next_image()
print 'framenumber:', frame_number, 'timestamp:', frame_timestamp

# read last frame
img, (frame_number, frame_timestamp) = store.get_image(store.frame_max)
print 'framenumber:', frame_number, 'timestamp:', frame_timestamp
```

# Install

Most of *IMGStore* dependencies are in the python package index (pypi),
with the honorable exception of *opencv*. If you have a python environment
with opencv already installed on it, you should be able to install IMGStore
with a command like:

```sh
pip install imgstore
```

## Install in Ubuntu 14.04

```sh
# install opencv, pandas and virtualenv
sudo apt-get install libopencv-dev python-opencv python-virtualenv python-pandas

# generate virtual env
virtualenv ~/.envs/imgstore --system-site-packages

# activate the virtual env
source ~/.envs/imgstore/bin/activate

# install imgstore
pip install imgstore
```


## Install in Mac OS X

Install [Homebrew](https://brew.sh/), you probably have to install Xcode first.

Then run:

```sh
PATH="/usr/local/bin:$PATH"

brew install python

brew tap homebrew/science
brew install opencv3 --with-ffmpeg

# follow the instructions at the end of the install
echo /usr/local/opt/opencv3/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/opencv3.pth

# install imgstore
pip install imgstore
```
