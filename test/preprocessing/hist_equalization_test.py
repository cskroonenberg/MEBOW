import cv2
import torchvision
from PIL import Image

import cv2
import torchvision
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Given an input image, process the image with histogram equalization and save it.')
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

    img_pil = Image.fromarray(img)

    eq_img = torchvision.transforms.functional.equalize(img_pil)

    eq_img.save(args['o'])

if __name__ == "__main__":
    main()

