"""The overlays in build_base blend one colour through a coverage mask, in place.

The mask must be indistinguishable from the layer compositing it replaced: draw the
same one-colour overlay both ways over the same busy base and require byte equality.
Overlapping shapes are the case that makes the mask non-obvious — inside one overlay
they must blend once, not once per shape.
"""
from PIL import Image, ImageDraw


def _busy_base(w, h):
    img = Image.new("RGBA", (w, h), (30, 60, 90, 255))
    d = ImageDraw.Draw(img)
    for x in range(0, w, 7):
        d.line([(x, 0), (x, h)], fill=(x % 256, (3 * x) % 256, (5 * x) % 256, 255))
    return img


def test_mask_blend_is_byte_identical_to_layer_compositing():
    color = (36, 160, 98, 90)
    shapes = [
        [(10, 10), (120, 20), (80, 130)],
        [(60, 15), (140, 60), (90, 110)],  # overlaps the first: must blend once
        [(150, 100), (199, 149), (150, 149)],
    ]

    via_layer = _busy_base(200, 150)
    layer = Image.new("RGBA", via_layer.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for pts in shapes:
        ld.polygon(pts, fill=color)
    via_layer = Image.alpha_composite(via_layer, layer)

    via_mask = _busy_base(200, 150)
    mask = Image.new("L", via_mask.size, 0)
    md = ImageDraw.Draw(mask)
    for pts in shapes:
        md.polygon(pts, fill=color[3])
    ImageDraw.Draw(via_mask).bitmap((0, 0), mask, fill=color[:3] + (255,))

    assert via_mask.tobytes() == via_layer.tobytes()
