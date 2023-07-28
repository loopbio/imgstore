IMGStore - Houses Your Video And Data
=====================================

Imgstore is a container for video frames and metadata. It allows efficient storage and seeking
through recordings from hours to weeks in duration. It supports compressed and uncompressed formats.

Imgstore allows reading (and writing) videos recorded with
loopbio's [Motif](http://loopbio.com/recording/) recording system.

## Introduction

### The Concept

Video data is broken into chunks, which can be individual video files `VideoImgStore`, or
a directory full of images `DirectoryImgStore`. The format of the chunks determines if the store is
compressed, uncompressed, lossless or lossy.

Along side the video data Imgstore can also record different types of metadata:
 * Recording (User) Metadata:  
   This includes information set at the time of recording, such as 'genotype', that is constant
   for the entire duration of recording.
 * Frame Metadata:  
   This is the `frame_number` and `frame_timestamp` for every frame in the imgstore
 * Extra Data:  
   This is DAQ and Audio data recorded by [Motif](http://loopbio.com/recording/) at a rate different, and
   often faster than, the video framerate.

See [Extracting Metadata](#extracting-metadata).

### Basic API

There are only a few public API entry points exposed (most operations are
done on `ImgStore` objects (see writing and reading examples below).

 * `new_for_filename(path)` - Open a store for reading
 * `new_for_format(format, path, **kwargs)`
    * Open a store for writing
    * You also need to pass `imgshape=` and `imgdtype`
    * Note: `imgshape` is the array shape, i.e. `(h,w,d)` and not `(w,h)`
 * `get_supported_formats()` - list supports formats (remember to test after install)
 * `extract_only_frame(path, frame_index)` - extract a single frame at given *index* from file

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

You can also add additional (JSON serialable) data at any time, and this will be stored
with a reference to the current `frame_number` so that it can be retrieved
and easily combined later.

```python
store.add_extra_data(temperature=42.5, humidity=12.4)
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
is any integer value associated with a single frame, and 'frame index', which is numbered
from 0 to the number of frames in the store. This difference is visible in the API with

```python
class ImgStore
    def get_image(self, frame_number, exact_only=True, frame_index=None):
        pass
```

where 'frame index' OR 'frame number' can be passed.

## Extracting Metadata

To get the Recording (user) metadata access the `ImgStore.user_metadata` property.

To get all the image metadata at once you can call `ImgStore.get_frame_metadata()`
which will return a dictionary containing all `frame_number` and `frame_time`stamps.

To retrieve a pandas DataFrame of all extra data and associated `frame_number`
and `frame_time`stamps call `ImgStore.get_extra_data()`

# Command line tools

Some simple tools for creating, converting and viewing imgstores are provided

* `imgstore-view /path/to/store`
  * view an imgstore
* `imgstore-save --format 'avc1/mp4' --source /path/to/input.mp4 /path/to/store/to/save`
  * `--source` if omitted will be the first webcam
* `imgstore-test`
  * run extensive tests to check opencv build has mp4 support and trustworthy encoding/decoding

# Install

*IMGStore* depends on reliable OpenCV builds, and built with mp4/h264 support for
writing mp4s.

You can install opencv from pip or conda if you wish, or on linux you can use the system apt-get
installed opencv if you create a virtual environment with `--system-site-packages`.

Once you have a python (virtual) environment with a recent and reliable OpenCV build,
you can install IMGStore from pip

`$ pip install imgstore`

After installing imgstore from any location, you should check it's tests pass to guarantee that
you have a trustworthy OpenCV version.

If you must install opencv from pip, such as for only reading imgstores, you can
use the following command to install the latest 4.8 version

`$ pip install "opencv-python < 4.9"`

## Post install testing

Test libraries are not installed by default. To do so, install pytest

`$ pip install "pytest < 8"`

You should always run the command `imgstore-test` after installing imgstore. If your
environment is working correctly you should see a lot of text printed, followed by the
text `==== 66 passed, ..... ======`

To test against the package without installing first, run `python -m pytest`

Note: by running pytest through it's python module interface, the interpreter adds `pwd` to
top of `PYTHONPATH`, as opposed to running tests through `py.test` which doesn't.

Note: if you recieve many failed tests with the error message 'The opencv backend does not actually have write support'
or 'Your opencv build does support writing this codec', this is __not an imgestore bug__ - it is a warning that
you have an OpenCV version that does not support _Writing_ h264 encoded videos. This is often the case on
windows.

**Even if some _write_ tests fail due to these issues, you can stil use the imgstore package to _read_ h264 encoded
video files.**

