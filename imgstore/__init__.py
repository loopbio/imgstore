from .stores import new_for_filename, new_for_format, extract_only_frame, get_supported_formats,\
    VideoImgStore, DirectoryImgStore
from .apps import main_test
from .util import ensure_color, ensure_grayscale

test = main_test

__version__ = '0.2.1'
