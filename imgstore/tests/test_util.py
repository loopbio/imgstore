import numpy as np

import cv2

import imgstore.util as ibu


def test_codec():
    c = ibu.ImageCodecProcessor()
    assert c.cv2_enum_to_code(cv2.COLOR_RGB2BGR) == 'cv_rgb'
    assert not c.check_code('foo')
    assert c.check_code('cv_bayerbg')

    bw = np.zeros((100, 100), np.uint8)
    color = c.convert_to_bgr(bw, 'cv_bayerbg')
    assert (color.ndim == 3) and (color.shape[2] == 3) and \
           (color.shape[0] == bw.shape[0]) and (color.shape[1] == bw.shape[1])

    c = ibu.ImageCodecProcessor.from_cv2_enum(cv2.COLOR_BayerRG2BGR)
    assert c.encoding == 'cv_bayerrg'

    c = ibu.ImageCodecProcessor.from_pylon_format('BayerBG')
    assert c.encoding == 'cv_bayerrg'
