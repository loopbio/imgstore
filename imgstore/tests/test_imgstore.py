from __future__ import print_function
import numpy as np
import numpy.testing as npt
import os.path
import shutil
import tempfile
import time
import itertools

import cv2
import pytest

from imgstore import stores
from imgstore.util import FourCC, ensure_color, ensure_grayscale
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


def new_framecode_video(dest, frame0, nframes):
    assert dest.endswith('.avi')
    cap = cv2.VideoWriter(dest, FourCC('M', 'J', 'P', 'G'), 25, (512, 512), isColor=True)
    for i in range(frame0, frame0+nframes):
        cap.write(ensure_color(encode_image(i, imgsize=512)))
    cap.release()
    return dest


def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def new_framecode_store(dest, frame0, nframes, format='png', chunksize=7):
    kwargs = dict(basedir=dest,
                  mode='w',
                  imgshape=(512, 512, 3),
                  imgdtype=np.uint8,
                  chunksize=chunksize,
                  format=format)

    d = stores.new_for_format(format, **kwargs)
    for i in range(frame0, frame0+nframes):
        frame = ensure_color(encode_image(i, imgsize=512))
        d.add_image(frame, i, time.time())
    d.close()

    return d, dest


@pytest.fixture
def loglevel_info():
    import logging
    logging.basicConfig(level=logging.INFO)


@pytest.fixture
def loglevel_debug():
    import logging
    logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def grey_image():
    cimg = cv2.imread(os.path.join(TEST_DATA_DIR, 'graffiti.png'), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(cimg, (1920, 1200))


@pytest.fixture
def graffiti():
    return cv2.imread(os.path.join(TEST_DATA_DIR, 'graffiti.png'), cv2.IMREAD_COLOR)


@pytest.mark.parametrize('fmt', stores.get_supported_formats())
def test_all(tmpdir, fmt):
    F = 21
    SZ = 512
    imgtype = 'color'

    tdir = tmpdir.strpath

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
                  chunksize=10,
                  metadata={'timezone': 'Europe/Austria'},
                  format=fmt)
    d = stores.new_for_format(fmt, **kwargs)

    fns = []
    frame_times = {}
    for i in range(F):
        t = time.time()
        d.add_image(_build_img(i), i, t)
        frame_times[i] = t
        fns.append(i)
    d.close()

    d = stores.new_for_filename(os.path.join(d.filename, stores.STORE_MD_FILENAME), mode='r')

    assert d.user_metadata['timezone'] == 'Europe/Austria'

    r = np.random.RandomState(42)
    r.shuffle(fns)
    for f in fns:
        img, (frame_number, frame_time) = d.get_image(frame_number=f)
        assert f == frame_number
        assert frame_times[f] == frame_time
        assert img.shape == orig_img.shape
        assert _decode_image(img) == f

    d.close()


@pytest.mark.parametrize('fmt', ('mjpeg', 'npy', 'h264/mkv', 'avc1/mp4'))
@pytest.mark.parametrize('imgtype', ('b&w', 'color'))
def test_outoforder(tmpdir,  fmt, imgtype):
    SZ = 512

    tdir = tmpdir.strpath

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
    assert os.path.isfile(os.path.join(d.filename, stores.STORE_LOCK_FILENAME))

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
    assert not os.path.isfile(os.path.join(d.filename, stores.STORE_LOCK_FILENAME))

    d = stores.new_for_filename(os.path.join(d.filename, stores.STORE_MD_FILENAME))

    # read mode doesnt create lock file
    assert not os.path.isfile(os.path.join(d.filename, stores.STORE_LOCK_FILENAME))

    assert d.image_shape == orig_img.shape
    assert d.has_extra_data
    assert d.frame_min == 0
    assert d.frame_max == F

    r = np.random.RandomState(42)
    r.shuffle(fns)
    for f in fns:
        img, (frame_number, frame_time) = d.get_image(frame_number=f)
        assert d.frame_number == f
        assert f == frame_number
        assert frame_times[f] == frame_time
        assert img.shape == orig_img.shape
        assert _decode_image(img) == f

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


@pytest.mark.xfail(strict=False)
def test_create_and_times(loglevel_debug, tmpdir):
    import pytz

    tdir = tmpdir.strpath

    # check we always get a UTC timesone
    # does not have created_utc in metadata
    d = stores.new_for_filename(os.path.join(TEST_DATA_DIR, 'store_mp4', 'metadata.yaml'))
    c_utc, tz = d.created
    assert c_utc.tzinfo is not None
    assert c_utc.tzinfo == pytz.utc

    store, _ = new_framecode_store(dest=tdir, frame0=0, nframes=1)
    assert store.uuid is not None

    store2 = stores.new_for_filename(store.full_path)
    assert store2.uuid == store.uuid
    assert store2.created == store.created

    c_utc, tz = store2.created
    assert c_utc.tzinfo is not None
    assert c_utc.tzinfo == pytz.utc


def test_testencode_decode():
    L = 16
    SZ = 512

    for i in (0, 1, 13, 127, 255, 34385, (2 ** L) - 1):
        img = encode_image(num=i, nbits=L, imgsize=SZ)
        v = decode_image(img, nbits=L, imgsize=SZ)
        assert v == i


@pytest.mark.parametrize('fmt', ('npy', 'mjpeg', 'h264/mkv', 'avc1/mp4'))
def test_extract_only(loglevel_debug, fmt, tmpdir):
    tdir = tmpdir.strpath

    d, _ = new_framecode_store(dest=tdir,
                               frame0=57, nframes=20, format=fmt)

    for i, fn in enumerate(range(57, 57+20)):
        img = stores.extract_only_frame(d.full_path, i)
        assert decode_image(ensure_color(img)[:, :, 0]) == fn


def test_extract_only_motif(loglevel_debug):
    full_path = os.path.join(TEST_DATA_DIR, 'store_mp4', 'metadata.yaml')
    img = stores.extract_only_frame(full_path, 42)
    assert decode_image(ensure_color(img)[:, :, 0]) == 42

    basedir = os.path.join(TEST_DATA_DIR, 'store_mp4')
    img = stores.extract_only_frame(basedir, 42)
    assert decode_image(ensure_color(img)[:, :, 0]) == 42


def test_manual_assembly(loglevel_debug, tmpdir):
    tdir = tmpdir.strpath

    a = new_framecode_video(dest=os.path.join(tdir, 'a.avi'),
                            frame0=0, nframes=10)

    a_fns = list(range(0, 10))  # [0 ... 9]

    b = new_framecode_video(dest=os.path.join(tdir, 'b.avi'),
                            frame0=57, nframes=20)

    b_fns = list(range(57, 57 + 20))  # [57 ... 76]

    dest = os.path.join(tdir, 'store')
    store = stores.VideoImgStore(basedir=dest,
                                 mode='w',
                                 imgshape=(512, 512, 3),
                                 imgdtype=np.uint8,
                                 chunksize=100,
                                 format='mjpeg')
    store.empty()

    store.insert_chunk(a, a_fns, 1000.0 * np.asarray(a_fns))
    store.insert_chunk(b, b_fns, 1000.0 * np.asarray(b_fns))
    store.close()

    store = stores.new_for_filename(store.full_path)

    fns = list(itertools.chain.from_iterable((a_fns, b_fns)))
    r = np.random.RandomState(42)
    r.shuffle(fns)

    for f in fns:
        img, (frame_number, frame_time) = store.get_image(frame_number=f, exact_only=True)
        assert frame_number == f
        assert frame_time == (1000.0 * f)
        assert decode_image(img[:, :, 0]) == f

    store.close()


def test_store_frame_metadata(tmpdir):
    SZ = 512

    tdir = tmpdir.strpath

    d = stores.DirectoryImgStore(basedir=tdir,
                                 mode='w',
                                 imgshape=(SZ, SZ),
                                 imgdtype=np.uint8,
                                 chunksize=5,
                                 format='npy')

    r = np.random.RandomState(42)

    N = 100

    fns = []
    frame_times = {}
    for i in range(N):

        img = encode_image(num=i, imgsize=SZ)

        # skip some frames
        if r.rand() < 0.3:  # skip 30% of frames
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


@pytest.mark.parametrize("seek", [True, False])
def test_videoimgstore_mp4(seek):
    L = 16
    SZ = 512

    d = stores.new_for_filename(os.path.join(TEST_DATA_DIR, 'store_mp4', 'metadata.yaml'),
                                seek=seek)
    assert d.frame_max == 178
    assert d.frame_min == 0
    assert d.chunks == [0, 1]

    for i in range(3):
        img, (_frame_number, _frame_timestamp) = d.get_next_image()
        assert img.shape == (SZ, SZ)
        assert _frame_number == i
        assert decode_image(img, nbits=L, imgsize=SZ) == i

    fns = list(range(0, 178))
    r = np.random.RandomState(42)
    r.shuffle(fns)

    for i in fns:
        img, (_frame_number, _frame_timestamp) = d.get_image(i)
        assert img.shape == (SZ, SZ)
        assert _frame_number == i
        assert decode_image(img, nbits=L, imgsize=SZ) == i


@pytest.mark.parametrize('chunksize', (2, 3, 10))
def test_reindex_to_zero(loglevel_debug, tmpdir, grey_image, chunksize):
    tdir = tmpdir.strpath

    fmt = 'mjpeg'
    kwargs = dict(basedir=tdir,
                  mode='w',
                  imgshape=grey_image.shape,
                  imgdtype=grey_image.dtype,
                  chunksize=chunksize,
                  metadata={'timezone': 'Europe/Austria'},
                  format=fmt)

    d = stores.new_for_format(fmt, **kwargs)

    d.add_image(grey_image, 3, time.time())
    d.add_extra_data(ofn=3)
    d.add_image(grey_image, 4, time.time())
    d.add_extra_data(ofn=4)
    d.add_image(grey_image, 0, time.time())
    d.add_image(grey_image, 1, time.time())
    d.add_image(grey_image, 2, time.time())
    d.add_image(grey_image, 3, time.time())
    d.add_image(grey_image, 4, time.time())
    d.add_extra_data(ofn=42)
    d.close()

    s = stores.new_for_filename(d.full_path, mode='r')
    npt.assert_array_equal(s.get_frame_metadata()['frame_number'], [3, 4, 0, 1, 2, 3, 4])

    df = s.get_extra_data()
    assert len(df) == 3
    assert sorted(df['ofn'].tolist()) == [3, 4, 42]
    assert sorted(df['frame_index'].tolist()) == [0, 1, 6]
    assert sorted(df['frame_number'].tolist()) == [3, 4, 4]

    s.reindex()
    s.close()

    j = stores.new_for_filename(s.full_path, mode='r')
    npt.assert_array_equal(j.get_frame_metadata()['frame_number'], [-2, -1, 0, 1, 2, 3, 4])

    df = j.get_extra_data()
    assert len(df) == 3
    assert sorted(df['ofn'].tolist()) == [3, 4, 42]
    assert sorted(df['frame_index'].tolist()) == [0, 1, 6]
    assert sorted(df['frame_number'].tolist()) == [-2, -1, 4]


def test_reindex_impossible(tmpdir, grey_image):
    tdir = tmpdir.strpath

    fmt = 'mjpeg'
    kwargs = dict(basedir=tdir,
                  mode='w',
                  imgshape=grey_image.shape,
                  imgdtype=grey_image.dtype,
                  chunksize=3,
                  metadata={'timezone': 'Europe/Austria'},
                  format=fmt)

    d = stores.new_for_format(fmt, **kwargs)
    d.add_image(grey_image, 3, time.time())
    d.add_image(grey_image, 4, time.time())
    d.add_image(grey_image, 0, time.time())
    d.add_image(grey_image, 0, time.time())
    d.add_image(grey_image, 1, time.time())
    d.close()

    s = stores.new_for_filename(d.full_path, mode='r')
    with pytest.raises(ValueError):
        s.reindex()
    s.close()


@pytest.mark.parametrize("seek", [True, False])
@pytest.mark.parametrize("chunksize", [7, 20, 100])
@pytest.mark.parametrize("fmt", ['npy', 'mjpeg', 'h264/mkv', 'avc1/mp4'])
def test_seek_types(loglevel_debug, tmpdir, chunksize, fmt, seek):
    tdir = tmpdir.strpath

    def _decode_image(_img):
        return decode_image(_img[:, :, 0], imgsize=512)

    kwargs = dict(basedir=tdir,
                  mode='w',
                  imgshape=(512, 512, 3),
                  imgdtype=np.uint8,
                  chunksize=chunksize,
                  format=fmt)

    d = stores.new_for_format(fmt, **kwargs)

    skips = range(2030, 2070)
    for i in range(2000, 2100):
        if i in skips:
            continue
        d.add_image(ensure_color(encode_image(i, imgsize=512)), frame_number=i, frame_time=time.time())
    d.close()

    d = stores.new_for_filename(d.full_path, seek=seek)
    assert len(d) == (100 - len(skips))
    assert d.frame_min == 2000
    assert d.frame_max == 2099

    # check seeking works
    frame, (frame_number, frame_timestamp) = d.get_image(frame_number=2000, exact_only=True)
    assert frame_number == 2000
    assert _decode_image(frame) == 2000

    frame, (frame_number, frame_timestamp) = d.get_image(frame_number=2001, exact_only=True)
    assert frame_number == 2001
    assert _decode_image(frame) == 2001

    # read next frame
    frame, (frame_number, frame_timestamp) = d.get_next_image()
    assert frame_number == 2002
    assert _decode_image(frame) == 2002

    # seek missing frame
    with pytest.raises(ValueError):
        frame, (frame_number, frame_timestamp) = d.get_image(frame_number=skips[0], exact_only=True)

    # check seeking works by frame_index
    for i in range(10, 15):
        frame, (frame_number, frame_timestamp) = d.get_image(frame_number=None, exact_only=True, frame_index=i)
        assert frame_number == 2000 + i
        assert _decode_image(frame) == 2000 + i

    # check seeking works by frame index when non monotonically increasing
    frame, (frame_number, frame_timestamp) = d.get_image(frame_number=None, exact_only=True, frame_index=29)
    assert frame_number == 2029
    assert _decode_image(frame) == 2029

    frame, (frame_number, frame_timestamp) = d.get_image(frame_number=None, exact_only=True, frame_index=30)
    assert frame_number == 2070
    assert _decode_image(frame) == 2070

    frame, (frame_number, frame_timestamp) = d.get_image(frame_number=None, exact_only=True, frame_index=31)
    assert frame_number == 2071
    assert _decode_image(frame) == 2071

    frame, (frame_number, frame_timestamp) = d.get_image(frame_number=None, exact_only=True, frame_index=0)
    assert frame_number == 2000
    assert _decode_image(frame) == 2000

    frame_count = 100 - len(skips)
    assert d.frame_count == frame_count

    # last frame
    frame, (frame_number, frame_timestamp) = d.get_image(frame_number=None, exact_only=True,
                                                         frame_index=frame_count - 1)
    assert frame_number == 2099
    assert _decode_image(frame) == 2099

    with pytest.raises(EOFError):
        frame, (frame_number, frame_timestamp) = d.get_next_image()

    with pytest.raises(ValueError):
        frame, (frame_number, frame_timestamp) = d.get_image(frame_number=None, exact_only=True,
                                                             frame_index=frame_count)


def test_always_supported():
    fmts = stores.get_supported_formats()
    assert 'mjpeg/avi' in fmts
    assert 'npy' in fmts


@pytest.mark.parametrize('fmt', stores.VideoImgStore.supported_formats())
@pytest.mark.parametrize("seek", [True, False])
def test_videoseek_extensive(loglevel_debug, fmt, seek, tmpdir):
    F = 1000
    S_PCT = 0.4

    SZ = 32

    tdir = tmpdir.strpath

    def _build_img(num):
        return ensure_color(encode_image(num, imgsize=SZ))

    def _decode_image(img):
        return decode_image(img[:, :, 0], imgsize=SZ)

    orig_img = _build_img(0)
    kwargs = dict(basedir=tdir,
                  mode='w',
                  imgshape=orig_img.shape,
                  imgdtype=orig_img.dtype,
                  chunksize=F,  # one chunk to test seeking of video file
                  format=fmt)

    with stores.new_for_format(fmt, **kwargs) as d:
        for i in range(F):
            t = time.time()
            d.add_image(_build_img(i), i, t)

    r = np.random.RandomState(0)
    fns = r.randint(low=0, high=F, size=int(S_PCT * F))

    with stores.new_for_filename(tdir, seek=seek) as d:
        for f in fns:
            img, _ = d.get_image(frame_number=f)
            assert _decode_image(img) == f


def test_new_apis(tmpdir):
    p = os.path.join(TEST_DATA_DIR, 'store_mp4')
    fullp = os.path.join(p, stores.STORE_MD_FILENAME)

    # new_for_filename (read)

    # read pass both paths
    with pytest.raises(ValueError):
        stores.new_for_filename(p, basedir=p)

    # read pass no path
    with pytest.raises(ValueError):
        stores.new_for_filename(None)

    # read pass path=basedir
    d = stores.new_for_filename(p)
    assert d.full_path == fullp
    assert d.mode == 'r'

    # read pass path=fullpath
    d = stores.new_for_filename(fullp)
    assert d.full_path == fullp
    assert d.mode == 'r'

    # read pass basedir=basedir
    d = stores.new_for_filename(None, basedir=p)
    assert d.full_path == fullp
    assert d.mode == 'r'

    with pytest.raises(ValueError):
        # read pass basedir=fullpath
        d = stores.new_for_filename(None, basedir=fullp)

    # new_for_format (read)

    d = stores.new_for_format(fmt=None, path=None, mode='r', basedir=p)
    assert d.full_path == fullp
    assert d.mode == 'r'

    d = stores.new_for_format(fmt=None, path=p, mode='r')
    assert d.full_path == fullp
    assert d.mode == 'r'

    d = stores.new_for_format(fmt=None, path=fullp, mode='r')
    assert d.full_path == fullp
    assert d.mode == 'r'

    # == new_for_format (write)

    # read pass no path
    with pytest.raises(ValueError):
        stores.new_for_format(fmt=None, mode='r')

    # write pass path=basedir
    p = tmpdir.mkdir('a').strpath
    fullp = os.path.join(p, stores.STORE_MD_FILENAME)
    d = stores.new_for_format(fmt='npy', path=p, imgshape=(10, 10), imgdtype=np.uint8)
    assert d.full_path == fullp
    assert d.mode == 'w'
    d.close()

    # write pass path=fullpath
    p = tmpdir.mkdir('b').strpath
    fullp = os.path.join(p, stores.STORE_MD_FILENAME)
    d = stores.new_for_format(fmt='npy', path=fullp, imgshape=(10, 10), imgdtype=np.uint8)
    assert d.full_path == fullp
    assert d.mode == 'w'
    d.close()

    # write pass basedir=basedir (API/back compat)
    p = tmpdir.mkdir('c').strpath
    fullp = os.path.join(p, stores.STORE_MD_FILENAME)
    d = stores.new_for_format(fmt='npy', basedir=p, imgshape=(10, 10), imgdtype=np.uint8)
    assert d.full_path == fullp
    assert d.mode == 'w'
    d.close()

    with pytest.raises(ValueError):
        # write pass basedir=fullpath
        p = tmpdir.mkdir('d').strpath
        fullp = os.path.join(p, stores.STORE_MD_FILENAME)
        d = stores.new_for_format(fmt='npy', basedir=fullp, imgshape=(10, 10), imgdtype=np.uint8)


@pytest.mark.parametrize("fmt", ['npy', 'mjpeg', 'avc1/mp4', 'h264/mkv'])
def test_odd_sized(fmt, tmpdir):
    img = np.zeros((199, 199, 3), dtype=np.uint8)
    d = stores.new_for_format(fmt,
                              basedir=tmpdir.strpath,
                              imgshape=img.shape,
                              imgdtype=img.dtype)
    for i in range(10):
        d.add_image(img, i, 0.)
    d.close()

    d = stores.new_for_filename(d.full_path)
    _img, _ = d.get_next_image()

    assert _img.shape == d.image_shape


@pytest.mark.parametrize('chunksize', (2,100))
@pytest.mark.parametrize("fmt", ['npy', 'mjpeg'])
def test_framenmber_non_monotonic_with_wrap(tmpdir, chunksize, fmt):
    FNS = [6050, 6055, 6056, 0, 1, 2]

    s = stores.new_for_format(fmt,
                              basedir=tmpdir.strpath,
                              imgshape=(512, 512),
                              imgdtype=np.uint8,
                              chunksize=chunksize)
    for fn in FNS:
        s.add_image(encode_image(fn, imgsize=512), fn, time.time())
    s.close()

    d = stores.new_for_filename(s.full_path)
    assert d.frame_min == 0
    assert d.frame_max == 6056
    assert d.frame_count == 6

    for fn in FNS:
        frame, (frame_number, frame_timestamp) = d.get_next_image()
        assert frame.shape == (512, 512)
        assert frame_number == fn
        assert decode_image(frame) == fn

    with pytest.raises(EOFError):
        d.get_next_image()

    # iter in reversed to make it more interesting
    for fn in reversed(FNS):
        frame, (frame_number, frame_timestamp) = d.get_image(fn, exact_only=True, frame_index=None)
        assert frame_number == fn
        assert decode_image(frame) == fn

    for fn in reversed(FNS):
        idx = FNS.index(fn)
        frame, (frame_number, frame_timestamp) = d.get_image(frame_number=None, exact_only=True,
                                                             frame_index=idx)
        assert frame_number == fn
        assert decode_image(frame) == fn

    # seek to middle and then keep reading
    fn = 6055
    frame, (frame_number, frame_timestamp) = d.get_image(fn, exact_only=True, frame_index=None)
    assert frame_number == fn
    assert decode_image(frame) == fn

    fn = 6056
    frame, (frame_number, frame_timestamp) = d.get_next_image()
    assert frame_number == fn
    assert decode_image(frame) == fn

    fn = 0
    frame, (frame_number, frame_timestamp) = d.get_next_image()
    assert frame_number == fn
    assert decode_image(frame) == fn


@pytest.mark.parametrize("fmt", ['npy', 'mjpeg', 'avc1/mp4', 'h264/mkv'])
@pytest.mark.parametrize("nframes", [2, 3, 4])
@pytest.mark.parametrize("seek", [True, False], ids=['seek', 'noseek'])
def test_max_framenumber_behaviour(fmt, nframes, seek, tmpdir):
    _, path = new_framecode_store(dest=tmpdir.strpath,
                                  frame0=3, nframes=nframes, format=fmt, chunksize=3)

    store = stores.new_for_filename(path, seek=seek)
    assert store.frame_count == nframes

    img, (_fn, _) = store.get_image(frame_number=store.frame_max, exact_only=True)
    assert _fn == store.frame_max
    assert img is not None

    with pytest.raises(EOFError):
        store.get_next_framenumber()

    store.close()
    store = stores.new_for_filename(path, seek=seek)

    img, (_fn, _) = store.get_image(frame_number=store.frame_max - 1, exact_only=True)
    assert decode_image(img[:, :, 0]) == (store.frame_max - 1)
    img, (_fn, _) = store.get_image(frame_number=store.frame_max, exact_only=True)
    assert _fn == store.frame_max
    assert decode_image(img[:, :, 0]) == store.frame_max
    assert store.frame_number == store.frame_max

    with pytest.raises(EOFError):
        store.get_next_framenumber()

    store.close()
    store = stores.new_for_filename(path, seek=seek)

    for i in range(0, store.frame_count):
        img, (_fn, _) = store.get_image(frame_number=None, exact_only=True, frame_index=i)
        assert (_fn - 3) == i
        assert decode_image(img[:, :, 0]) == _fn

    with pytest.raises(ValueError):
        store.get_image(frame_number=None, exact_only=True, frame_index=store.frame_count)

