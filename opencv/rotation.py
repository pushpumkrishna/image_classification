import numpy as np
import argparse
import imutils
import cv2
from opencv.image_read import OpenCV

filepath = "/Users/pushpumkrishna/PycharmProjects/image_classification/sample_images/burger.jpg"
image = cv2.imread(filepath)
cv2.imshow("Original", image)

class Rotation(OpenCV):

    def __init__(self):
        super().__init__()



(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# scale=1.0 means the same dimensions of the image
# 2.0 the image would be doubled in size.
# a value of 0.5 halves the size of the image.

M = cv2.getRotationMatrix2D(center=center, angle=45, scale=1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 Degrees", rotated)
M = cv2.getRotationMatrix2D(center, -90, 1.0)

rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by -90 Degrees", rotated)
