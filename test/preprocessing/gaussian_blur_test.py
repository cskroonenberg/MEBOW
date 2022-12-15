import cv2
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Given an input image, process the image with gaussian_blur and save it.')
    # general
    parser.add_argument('input_image_path',
                        help='Path of image to be processed',
                        type=str)
    parser.add_argument('-o',
                        help='Path to save processed image. Default=out.jpg',
                        default='out.jpg',
                        type=str)

    args = parser.parse_args()

    return args

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

def main():
    args = vars(parse_args())
    img = cv2.imread(args['input_image_path'])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torchvision.transforms.ToTensor()(img)

    blur = torchvision.transforms.GaussianBlur(9, sigma=(0.1, 5.0))

    blurred_img = blur.forward(img_t)

    torchvision.utils.save_image(blurred_img, args['o'])

if __name__ == "__main__":
    main()