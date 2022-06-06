# import the necessary packages
from typing import Tuple

import numpy as np
import cv2
import os

from utils.constant import labels


class LinearReg:

    def __init__(self,
                 ):
        super(LinearReg, self).__init__()

    # @staticmethod
    def read_image(self,
                   ) -> Tuple[any, any]:
        os.chdir("/Users/pushpumkrishna/PycharmProjects/image_classification/")
        original = cv2.imread(self.filepath)
        image = cv2.resize(original, (32, 32)).flatten()

        return (original, image)

    @staticmethod
    def generate_weight():
        # be *learned* by our model, but for the sake of this example,
        # let’s use random values
        weight = np.random.randn(3, 3072)
        bias = np.random.randn(3)
        #
        return (weight, bias)

    def calculate_score(self,
                        ):
        original, image = self.read_image()
        weight, bias = LinearReg.generate_weight()

        scores = weight.dot(image) + bias

        return scores

    def result(self,
               filepath: str):

        self.filepath = filepath
        # loop over the scores + labels and display them
        score = self.calculate_score()
        for (label, score) in zip(labels, score):
            print("[INFO] {}: {:.2f}".format(label, score))

    # def display_result(self, ):
    #     # draw the label with the highest score on the image as our
    #     # prediction
    #     cv2.putText(self.original,
    #                 "Label: {}".format(labels[np.argmax(self.calculate_score())]),
    #                 (10, 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.9,
    #                 (0, 255, 0),
    #                 2)
    #     # display our input image
    #     # cv2.imshow("Image", original)
    #     # cv2.waitKey(0)


a = LinearReg()
# print(a.result("./animals/cats/cats_00001.jpg"))
a.result("./animals/cats/cats_00001.jpg")
# print(a)
