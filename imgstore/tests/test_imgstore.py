from __future__ import print_function
import numpy as np
import numpy.testing as npt
import os.path
import random
import shutil
import tempfile
import time

import cv2
import pytest

from imgstore import stores
from imgstore.tests import TEST_DATA_DIR


def encode_image(num, nbits=16, imgsize=512):
    # makes a square image that looks a bit like a barcode. colums of the matrix all 255
    # if the bit in the number is 1, 0 otherwise
    row = np.fromiter((255*int(c) for c in ('{0:0%db}' % nbits).format(num)), dtype=np.uint8)
    mat = np.tile(row, (nbits, 1))
    return cv2.resize(mat, dsize=(imgsize, imgsize), interpolation=cv2.INTER_NEAREST)


def decode_image(img, nbits=16, imgsize=512):
    h, w = img.shape[:2]
    assert (h == imgsize) and (w == imgsize)
    assert len(img.shape) == 2

    img = cv2.resize(img, dsize=(nbits, nbits), interpolation=cv2.INTER_NEAREST)
    row = (np.mean(img, axis=0) > 127).astype(np.uint8)
    bstr = ''.join(str(v) for v in row)
    return int(bstr, 2)


def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


@pytest.fixture
def grey_image():
    cimg = cv2.imread(os.path.join(TEST_DATA_DIR, 'graffiti.png'), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(cimg, (1920, 1200))


@pytest.fixture
def graffiti():
    return cv2.imread(os.path.join(TEST_DATA_DIR, 'graffiti.png'), cv2.IMREAD_COLOR)


@pytest.mark.parametrize('fmt', (pytest.mark.skipif(stores.bloscpack is None,
                                                    reason='bloscpack not installed')('bpk'),
                                 'mjpeg',
                                 'npy', 'tif', 'png', 'ppm', 'pgm', 'pbm', 'bmp', 'jpg'))
def test_imgstore(request, grey_image, fmt):

    # tdir = tempfile.mkdtemp(dir='/path/to/ssd/') for performance testing
    tdir = tempfile.mkdtemp()
    request.addfinalizer(lambda: shutil.rmtree(tdir))

    kwargs = dict(basedir=tdir,
                  mode='w',
                  imgshape=grey_image.shape,
                  imgdtype=grey_image.dtype,
                  chunksize=10,  # test a small chunksize so we can hit interesting edge cases
                  metadata={'timezone': 'Europe/Austria'},
                  format=fmt)

    d = stores.new_for_format(fmt, **kwargs)

    frame_times = {}

    F = 100

    t0 = time.time()
    for i in range(F):
        t = time.time()
        d.add_image(grey_image, i, t)
        frame_times[i] = t
    d.close()
    dt = time.time() - t0

    sz_bytes = get_size(tdir)
    cmp_ratio = abs(100 - (100.0 * ((F * float(grey_image.nbytes)) / sz_bytes)))

    orig_img = grey_image
    print("\n%s %s %dx%dpx@%s frames took %.1fs @ %.1f fps (size: %.1fMB, %.1f%% %scompression)" %
          (fmt, F,
           orig_img.shape[1], orig_img.shape[0], orig_img.dtype,
           dt, F / dt, sz_bytes / (1024 * 1024.),
           cmp_ratio,
           '' if d.lossless else 'LOSSY'))

    d = stores.new_for_filename(os.path.join(d.filename, stores.STORE_MD_FILENAME),
                                basedir=tdir, mode='r')

    assert d.user_metadata['timezone'] == 'Europe/Austria'

    for f in (0, 1, 12, 3, 27, 50, 5, 99):
        img, (frame_number, frame_time) = d.get_image(frame_number=f)
        assert f == frame_number
        assert frame_times[f] == frame_time
        assert img.shape == grey_image.shape

        if d.lossless:
            npt.assert_equal(grey_image, img)

    d.close()


@pytest.mark.parametrize('fmt', ('mjpeg',))
def test_videoimgstore(request, graffiti, fmt):

    tdir = tempfile.mkdtemp()
    request.addfinalizer(lambda: shutil.rmtree(tdir))

    orig_img = graffiti

    kwargs = dict(basedir=tdir,
                  mode='w',
                  imgshape=orig_img.shape,
                  imgdtype=orig_img.dtype,
                  chunksize=50,
                  format=fmt)

    d = stores.new_for_format(fmt, **kwargs)

    F = 100

    t0 = time.time()

    frame_times = {}
    frame_extra = {}
    for i in range(F):
        t = time.time()
        d.add_image(orig_img.copy(), i, t)
        frame_times[i] = t

        d.add_extra_data(N=i)
        frame_extra[i] = dict(N=i)

    d.close()
    dt = time.time() - t0

    sz_bytes = get_size(tdir)
    cmp_ratio = abs(100 - (100.0 * ((F * float(orig_img.nbytes)) / sz_bytes)))

    print("\n%s %s %dx%dpx@%s frames took %.1fs @ %.1f fps (size: %.1fMB, %.1f%% %scompression)" %
          (fmt, F,
           orig_img.shape[1], orig_img.shape[0], orig_img.dtype,
           dt, F/dt, sz_bytes/(1024*1024.),
           cmp_ratio,
           '' if d.lossless else 'LOSSY '))

    d = stores.new_for_filename(os.path.join(d.filename, stores.STORE_MD_FILENAME),
                                basedir=tdir, mode='r')

    for f in range(F):
        img, (frame_number, frame_time) = d.get_image(frame_number=f)
        assert f == frame_number
        assert frame_times[f] == frame_time
        assert img.shape == orig_img.shape

    md = d.get_frame_metadata()
    npt.assert_array_equal(list(frame_times.values()), md['frame_time'])
    npt.assert_array_equal(list(frame_times.keys()), md['frame_number'])

    df = d.get_extra_data()
    assert len(df) == len(frame_extra)
    df.set_index('frame_number', inplace=True, verify_integrity=True)
    for _i, _v in frame_extra.items():
        _row = df.loc[_i]
        assert _row['N'] == _i

    d.close()


@pytest.mark.parametrize('fmt', ('mjpeg', 'npy'))
@pytest.mark.parametrize('imgtype', ('b&w', 'color'))
def test_imgstore_outoforder(request,  fmt, imgtype):
    SZ = 512

    tdir = tempfile.mkdtemp()
    request.addfinalizer(lambda: shutil.rmtree(tdir))

    def _build_img(num):
        img = encode_image(num, imgsize=SZ)
        if imgtype == 'color':
            return np.dstack((img, img, img))
        return img

    def _decode_image(img):
        if imgtype == 'color':
            return decode_image(img[:, :, 0], imgsize=SZ)
        return decode_image(img, imgsize=SZ)

    orig_img = _build_img(0)
    kwargs = dict(basedir=tdir,
                  mode='w',
                  imgshape=orig_img.shape,
                  imgdtype=orig_img.dtype,
                  chunksize=5,
                  format=fmt)

    d = stores.new_for_format(fmt, **kwargs)

    assert d.image_shape == orig_img.shape

    F = 50
    assert F > 20+1
    assert F < 255

    fns = []
    frame_times = {}
    frame_extra = {}
    for i in range(5):
        t = time.time()

        # store the actual frame number
        img = _build_img(i)

        d.add_image(img, i, t)
        fns.append(i)
        frame_times[i] = t

        edat = dict(foo=123, N=i)
        d.add_extra_data(**edat)
        frame_extra[i] = edat

    for i in (6, 8, 9, 11):
        t = time.time()
        img = _build_img(i)
        d.add_image(img, i, t)
        fns.append(i)
        frame_times[i] = t

        edat = dict(foo=123, N=i)
        d.add_extra_data(**edat)
        frame_extra[i] = edat

    for i in range(20, 15, -1):
        t = time.time()
        img = _build_img(i)
        d.add_image(img, i, t)
        fns.append(i)
        frame_times[i] = t
    for i in range(20+1, F+1):
        t = time.time()
        img = _build_img(i)
        d.add_image(img, i, t)
        fns.append(i)
        frame_times[i] = t

    d.close()

    d = stores.new_for_filename(os.path.join(d.filename, stores.STORE_MD_FILENAME))

    assert d.image_shape == orig_img.shape

    assert d.frame_min == 0
    assert d.frame_max == F

    random.shuffle(fns)
    for f in fns:
        truth_img = _build_img(f)

        img, (frame_number, frame_time) = d.get_image(frame_number=f)
        assert d.frame_number == f
        assert f == frame_number
        assert frame_times[f] == frame_time
        assert img.shape == truth_img.shape

        truth = _decode_image(truth_img)
        this = _decode_image(img)
        assert truth == this

    md = d.get_frame_metadata()
    npt.assert_array_equal(sorted([frame_times[f] for f in fns]), sorted(md['frame_time']))
    npt.assert_array_equal(sorted(fns), sorted(md['frame_number']))

    df = d.get_extra_data()
    assert len(df) == len(frame_extra)

    df.set_index('frame_number', inplace=True, verify_integrity=True)
    for _i, _v in frame_extra.items():
        _row = df.loc[_i]
        assert _row['foo'] == _v['foo']
        assert _row['N'] == _i

    sz_bytes = get_size(tdir)
    cmp_ratio = abs(100 - (100.0 * ((len(fns) * float(orig_img.nbytes)) / sz_bytes)))

    print("\n%s %s %dx%dpx@%s frames (size: %.1fMB, %.1f%% %scompression)" % (
        fmt, F,
        orig_img.shape[1], orig_img.shape[0], orig_img.dtype,
        sz_bytes / (1024 * 1024.),
        cmp_ratio,
        '' if d.lossless else 'LOSSY '))


def test_testencode_decode():
    L = 16
    SZ = 512

    for i in (0, 1, 13, 127, 255, 34385, (2 ** L) - 1):
        img = encode_image(num=i, nbits=L, imgsize=SZ)
        v = decode_image(img, nbits=L, imgsize=SZ)
        assert v == i


def test_store_frame_metadata(request):
    SZ = 512

    tdir = tempfile.mkdtemp()
    request.addfinalizer(lambda: shutil.rmtree(tdir))

    d = stores.DirectoryImgStore(basedir=tdir,
                                 mode='w',
                                 imgshape=(SZ, SZ),
                                 imgdtype=np.uint8,
                                 chunksize=5,
                                 format='npy')

    N = 100

    fns = []
    frame_times = {}
    for i in range(N):

        img = encode_image(num=i, imgsize=SZ)

        # skip some frames
        if random.random() < 0.3:  # skip 30% of frames
            continue

        frame_number = i
        frame_time = time.time()
        d.add_image(img, frame_number, frame_time)
        fns.append(frame_number)
        frame_times[i] = frame_time

    d.close()

    assert d.frame_count == len(fns)

    d = stores.DirectoryImgStore(basedir=tdir, mode='r')

    assert d.frame_count == len(fns)

    assert d.image_shape == (SZ, SZ)

    assert d.frame_min == fns[0]
    assert d.frame_max == fns[-1]

    for f in range(N):
        img, (frame_number, frame_time) = d.get_image(frame_number=f, exact_only=False)

        if f in frame_times:
            # we skipped this frame
            assert d.frame_number == f
            assert f == frame_number
            assert frame_times[f] == frame_time

        assert img.shape == (SZ, SZ)

        sfn = decode_image(img, imgsize=SZ)
        assert int(sfn) == frame_number

    d = stores.DirectoryImgStore(basedir=tdir, mode='r')

    fniter = iter(fns)
    while True:
        try:
            img, (frame_number, frame_time) = d.get_next_image()
            assert frame_number == next(fniter)
        except EOFError:
            break

    d.close()


def test_videoimgstore_mp4():
    L = 16
    SZ = 512

    d = stores.new_for_filename(os.path.join(TEST_DATA_DIR, 'store_mp4', 'metadata.yaml'))
    assert d.frame_max == 178
    assert d.frame_min == 0
    assert d.chunks == [0, 1]

    for i in range(3):
        img, (_frame_number, _frame_timestamp) = d.get_next_image()
        assert img.shape == (SZ, SZ)
        assert _frame_number == i
        assert decode_image(img, nbits=L, imgsize=SZ) == i

    for i in (7, 57, 98, 12, 168):
        img, (_frame_number, _frame_timestamp) = d.get_image(i)
        assert img.shape == (SZ, SZ)
        assert _frame_number == i
        assert decode_image(img, nbits=L, imgsize=SZ) == i
