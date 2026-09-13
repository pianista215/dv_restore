"""Composicion de imagenes para revisar a ojo: comparativas y hojas de contacto."""

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None

BG = (24, 24, 28)
FG = (235, 235, 235)


def _pil(img):
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    return Image.fromarray(img.astype(np.uint8))


def strip(panels, labels, scale=0.5, title=None, cols=None):
    """Rejilla de imagenes rotuladas."""
    if Image is None:
        raise RuntimeError("hace falta Pillow")
    ims = [_pil(p) for p in panels]
    w, h = ims[0].size
    w, h = int(w * scale), int(h * scale)
    ims = [im.resize((w, h), Image.BILINEAR) for im in ims]
    cols = cols or len(ims)
    rows = (len(ims) + cols - 1) // cols
    pad, lab, top = 6, 18, (26 if title else 0)
    W = cols * w + (cols + 1) * pad
    H = top + rows * (h + lab) + (rows + 1) * pad
    out = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(out)
    if title:
        d.text((pad, 6), title, fill=FG)
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        x = pad + c * (w + pad)
        y = top + pad + r * (h + lab + pad)
        out.paste(im, (x, y))
        d.text((x, y + h + 3), labels[k][:90], fill=FG)
    return np.array(out)


def contact(images, labels, cols=10, scale=0.16, title=None):
    return strip(images, labels, scale=scale, title=title, cols=cols)


def save(img, path):
    _pil(img).save(path)
