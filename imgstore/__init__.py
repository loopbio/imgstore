from .stores import new_for_filename, new_for_format, extract_only_frame, get_supported_formats,\
    VideoImgStore, DirectoryImgStore
from .apps import main_test
from .util import ensure_color, ensure_grayscale

test = main_test

# use only single quotes here (you can write beta versions like -beta4 etc)
__version__ = '0.3.7'
