from scipy import ndimage

import torch
from torch import Tensor

#use this with an image passed in (usually with image.imread(...))
#this is the least painful implementation without having to use
#a manually defined kernel and going through the image by hand

class UnsharpMasking(torch.nn.Module):
    """Apply gaussian blur to an image, and subtract the result from the original to sharpen at
	    the cost of continuity. 

    Returns:
        PIL Image or Tensor: Unsharp masking version of the input image.

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
