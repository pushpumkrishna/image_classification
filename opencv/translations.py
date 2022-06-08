import numpy as np
import argparse
import imutils
import cv2
from opencv.image_read import OpenCV

filepath = "/Users/pushpumkrishna/PycharmProjects/image_classification/sample_images/elephant.jpg"
oc = OpenCV()
image = oc.load_image_cv2(filepath=filepath)
cv2.imshow("Original", image)
cv2.waitKey(0)


class Translation:

    def __init__(self):
        pass

    def shift_image(self):
        """

        :return:
        """

        # Shift Right and Up
        matrix_1 = np.float32([[1, 0, 25],  # shift image to Right (Positive Value)
                               [0, 1, 50]])    # shift image to Up (Positive Value)

        shifted = cv2.warpAffine(image, matrix_1, (image.shape[1], image.shape[0]))
        cv2.imshow("Shifted Down and Right", shifted)
        cv2.waitKey(0)

        # Shift Left and down
        matrix_2 = np.float32([[1, 0, -50],  # shift image to Left (Negative Value)
                               [0, 1, -90]])   # shift image to Down (Negative Value)

        shifted = cv2.warpAffine(image, matrix_2, (image.shape[1], image.shape[0]))
        cv2.imshow("Shifted Up and Left", shifted)
        cv2.waitKey(0)


