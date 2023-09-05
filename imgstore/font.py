import cv2
import numpy as np

FONT_POS_TOP = 0
FONT_POS_BOTTOM = 1
FONT_POS_TOP_LEFT = 2

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX


# todo: opencv 3.3? added getFontScaleForHeight which can help to auto-scale text


def fit_text(txt, img, pos=FONT_POS_TOP, color=(255, 255, 255), name=cv2.FONT_HERSHEY_SIMPLEX, only_measure=False,
             img_shape=None):
    if pos not in {FONT_POS_BOTTOM, FONT_POS_TOP}:
        raise ValueError('invalid position')

    if (img is None) and (img_shape is None):
        raise ValueError('img or img_shape must be provided')

    def _measure_string(_scale, _thickness, _pad=1):
        # getTextSize under-measures height
        (w, h), _ = cv2.getTextSize(txt, name, _scale, _thickness)
        return _pad + w, _pad + h + (int(_scale + 0.5) * 1)

    tol = 0.90  # text should be 90% of the image width

    if img_shape is None:
        img_shape = img.shape[:2]

    ih, iw = img_shape
    tw, th = iw, ih
    scale = thickness = 1

    # keep bisecting a scale until the text it >= 90% of the image width
    tries = 0
    pct = (tw / float(iw))
    while (tries < 20) and (not ((pct > tol) and (tw < iw))):
        # shitty binary-ish search
        # increase by 100% if smaller, decrease by 25% if larger, so as to approach the limit
        scale *= (2.0 if pct < tol else 0.75)
        thickness = int(max(1, scale // 2))
        tw, th = _measure_string(scale, thickness)
        pct = (tw / float(iw))
        tries += 1

    if pos == FONT_POS_TOP:
        pos = (1, th + 3)
    elif pos == FONT_POS_BOTTOM:
        pos = (1, ih - 3)

    if not only_measure:
        cv2.putText(img, txt, pos, name, scale, color, thickness)

    return scale, thickness


def measure_string(s, name, scale, thickness):
    # getTextSize under-measures height
    (w, h), _ = cv2.getTextSize(s, name, scale, thickness)
    return w, h + (int(scale + 0.5) * 1)


class FontRenderer(object):

    def __init__(self, name, scale, thickness):
        self.name = name
        self.scale = scale
        self.thickness = thickness

    def measure_string(self, s):
        return measure_string(s, self.name, self.scale, self.thickness)

    def write_string(self, s, pos, color, img=None, bgcolor=None):
        """
        write the string onto the image.

        If no image supplied, create one that is cropped to the size of the text, and return it.
        The local origin is bottom left of text string, image coordinates are TL=0,0 BR=W,H

        Args:
            s: string
            pos: (x,y), or FONT_POS_TOP_LEFT
                bottom left coordinate of string
            color: (b,g,r)
            img: array

        Returns: the image written onto
        """
        w = h = None

        if pos == FONT_POS_TOP_LEFT:
            w, h = self.measure_string(s)
            pos = (0, h)

        # make a large enough image
        if img is None:
            if w is None:
                w, h = self.measure_string(s)
            img = np.zeros((h + pos[1] + 1, w + pos[0] + 1, 3), np.uint8)
            pos = pos[0], pos[1] + h

        if bgcolor is not None:
            if w is None:
                w, h = self.measure_string(s)
            x, y = pos
            pt1 = (x, y)
            pt2 = (x + w, y - h)
            cv2.rectangle(img, pt1, pt2, bgcolor, -1)

        cv2.putText(img, s, pos,
                    self.name, self.scale, color, self.thickness)

        return img

    def write_paragraph(self, s, pos, color, img=None):
        w = 0

        if pos == FONT_POS_TOP_LEFT:
            pos = (0, 0)

        x, y = pos
        lines = []
        # measure and layout all lines
        for l in s.split('\n'):
            _w, h = self.measure_string(l)
            w = max(_w, w)
            lines.append((l, (x, y + h)))
            y += int(1.2 * h)

        # make a large enough image
        if img is None:
            img = np.zeros((y, w + x, 3), np.uint8)

        for l, pos in lines:
            self.write_string(l, pos, color, img=img)

        return img


class FontRendererStatusBar(FontRenderer):
    POS_TOP = FONT_POS_TOP
    POS_BOTTOM = FONT_POS_BOTTOM
    POS_TOP_LEFT = FONT_POS_TOP_LEFT

    def __init__(self, name, scale=1, thickness=1):
        super(FontRendererStatusBar, self).__init__(name, scale, thickness)

        self._pos = None
        # x,y,w,h (w,h include padding)
        self._layout_l = None
        self._layout_r = None
        self._layout_box = None

    @classmethod
    def new_for_image(cls, img, font_name=DEFAULT_FONT, pos=POS_BOTTOM, **layout_and_init_args):
        h, w = img.shape[:2]

        # check if all the text can in principle fit...
        txt = ' '  # one space between left and right required
        txt += layout_and_init_args.get('ltext', '')
        txt += layout_and_init_args.get('rtext', '')
        scale = layout_and_init_args.get('scale', 1)
        thickness = layout_and_init_args.get('thickness', 1)
        padh = layout_and_init_args.get('padh', 12)
        _w, _h = measure_string(txt, font_name, scale, thickness)
        if (_w + (2 * padh)) > w:
            layout_and_init_args['scale'], layout_and_init_args['thickness'] = fit_text(txt, img=None,
                                                                                        name=font_name,
                                                                                        only_measure=True,
                                                                                        img_shape=(h, w - (2 * padh)))

        return FontRendererStatusBar.new_for_image_size((w, h),
                                                        font_name,
                                                        pos,
                                                        **layout_and_init_args)

    @classmethod
    def new_for_image_size(cls, frame_size, font_name=DEFAULT_FONT, pos=POS_BOTTOM, **layout_and_init_args):
        init_args = {}
        try:
            init_args['scale'] = layout_and_init_args.pop('scale')
        except KeyError:
            pass
        try:
            init_args['thickness'] = layout_and_init_args.pop('thickness')
        except KeyError:
            pass

        renderer = cls(font_name, **init_args)
        renderer.layout(pos, frame_size, **layout_and_init_args)

        return renderer

    def layout(self, pos, winsize, ltext="", rtext="", padv=4, padh=12):
        w, h = winsize

        ltw, lth = self.measure_string(ltext) if ltext else (0, 0)
        rtw, rth = self.measure_string(rtext) if rtext else (0, 0)

        if pos == FONT_POS_BOTTOM:
            if ltext:
                self._layout_l = (padh, h - padv, ltw, lth)
            if rtext:
                self._layout_r = (w - rtw - padh, h - padv, rtw, rth)
            bh = max(lth + (2 * padv), rth + (2 * padv))
            self._layout_box = (0, h - bh, w, bh)
        elif pos == FONT_POS_TOP:
            if ltext:
                self._layout_l = (padh, lth + padv, ltw, lth)
            if rtext:
                self._layout_r = (w - rtw - padh, rth + padv, rtw, rth)
            self._layout_box = (0, 0, w, max(lth + (2 * padv), rth + (2 * padv)))

    def render(self, img, ltext="", lcolor=(255, 255, 255), rtext="", rcolor=(255, 255, 255), bgcolor=None):
        if bgcolor:
            x, y, w, h = self._layout_box
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(img, pt1, pt2, bgcolor, -1)

        if ltext and self._layout_l:
            cv2.putText(img, ltext,
                        (self._layout_l[0], self._layout_l[1]),
                        self.name, self.scale, lcolor, self.thickness)
        if rtext and self._layout_r:
            cv2.putText(img, rtext,
                        (self._layout_r[0], self._layout_r[1]),
                        self.name, self.scale, rcolor, self.thickness)

    def extract(self, img):
        l, r = None, None
        x, y, w, h = self._layout_l if self._layout_l else (0, 0, 0, 0)
        if w:
            l = img[y - h:y + 1, x:x + w + 1, :]
        x, y, w, h = self._layout_r if self._layout_r else (0, 0, 0, 0)
        if w:
            r = img[y - h:y + 1, x:x + w + 1, :]
        return l, r
