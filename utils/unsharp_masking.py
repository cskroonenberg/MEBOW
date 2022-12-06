from scipy import ndimage

import torch
from torch import Tensor

#use this with an image passed in (usually with image.imread(...))
#this is the least painful implementation without having to use
#a manually defined kernel and going through the image by hand

class UnsharpMasking(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, sigma=5):
        super().__init__()
        self.sigma = sigma

    def forward(self, img: Tensor) -> Tensor:
        """Return a preprocessed image based on the unsharp masking
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        # subtract a gaussian filter/blur from the image
        return img - ndimage.gaussian_filter(img, self.sigma)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(sigma={self.sigma})"
        return s
