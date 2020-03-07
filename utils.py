import torchvision.transforms.functional as Fv

class FixedHeightResize(object):
    """
    from https://github.com/pytorch/vision/issues/908
    """
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        size = (self.height, self._calc_new_width(img))
        return Fv.resize(img, size)

    def _calc_new_width(self, img):
        old_width, old_height = img.size
        aspect_ratio = old_width / old_height
        return round(self.height * aspect_ratio)

