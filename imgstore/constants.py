try:
    # python 3
    # noinspection PyProtectedMember
    from subprocess import DEVNULL
    # noinspection PyShadowingBuiltins
    xrange = range
except ImportError:
    # python 2
    import os
    DEVNULL = open(os.devnull, 'r+b')

STORE_MD_KEY = '__store'
STORE_MD_FILENAME = 'metadata.yaml'
STORE_LOCK_FILENAME = '.lock'

EXTRA_DATA_FILE_EXTENSIONS = ('.extra.json', '.extra_data.json', '.extra_data.h5')

FRAME_MD = ('frame_number', 'frame_time')

