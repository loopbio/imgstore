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

store = imgstore.new_for_format('npy',  # numpy format (uncompressed raw image frames)
                                mode='w', basedir='mystore',
                                imgshape=blank_image.shape, imgdtype=blank_image.dtype,
                                chunksize=1000)  # 1000 files per chunk (directory)

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

## Extracting frames: frame index vs frame number

Stores maintain two separate and distinct concepts, 'frame number', which
is any integer vallue associated with a single frame, and 'frame index', which is numbered
from 0 to the number of frames in the store. This difference is visible in the API with

```python
class ImgStore
    def get_image(self, frame_number, exact_only=True, frame_index=None):
        pass
```

where 'frame index' OR 'frame number' can be passed.


# Command line tools

Some simple tools for creating, converting and viewing imgstores are provided

* `imgstore-view /path/to/store`
  view an imgstore
* `imgstore-save --format 'avc1/mp4' --source /path/to/input.mp4 /path/to/store/to/save`
 * `--source` if omitted will be the first webcam
* `imgstore-test`
  run extensive tests to check opencv build has mp4 support and trustworthy encoding/decoding


# Install

*IMGStore* depends on reliable OpenCV builds, and built with mp4/h264 support for
writing mp4s. Loopbio provides [reliable conda OpenCV builds](http://blog.loopbio.com/conda-packages.html)
in our conda channel, and we recommend using these.

We also recommend installing *IMGStore* from conda

' $ conda install imgstore`

After installing imgstore from any location, you should check it's tests pass to guarentee that
you have a trustworthy OpenCV version

## Installing from source

 * git clone this repository
 * `conda env create -f environment.yaml`

## Installing from pypi

We recommend installing *IMGStore* and its dependencies using the conda package manager, however
it is possible to install using pip. We recommend first creating a new virtual environment 

```sh
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

## Post install testing

You should always run the command `imgstore-test` after installing imgstore. If your
environment is working correctly you should see a lot of text printed, followed by the
text `==== 66 passed, ..... ======`
