from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class OpenCV:

    def __init__(self,
                 ):

        pass

    def load_image_cv2(self,
                       filepath: str):
        """
        TO load image from local disk
        :return: Image
        """

        self.filepath = filepath
        self.image = cv2.imread(self.filepath)

        print("Image width: {} pixels".format(self.image.shape[1]))
        print("Image height: {} pixels".format(self.image.shape[0]))
        print("Image channels: {}".format(self.image.shape[2]))

        # # display image
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)  # stays on screen unless enter is press
        # # cv2.destroyAllWindows()

        return self.image

    def load_image_matplotlib(self):
        """

        :return:
        """

        image = mpimg.imread(self.filepath)
        # plt.imshow(image)
        # plt.show()

        return image

    def write_image(self,
                    newImagePath: str,
                    image):
        """

        :param newImagePath:
        :param image
        :return: jpg image
        """

        # image = self.load_image_from_disk(self.filepath)

        return cv2.imwrite(newImagePath, image)

    def access_pixels(self,
                  x: int,
                  y: int):
        """

        :return:
        """

        (b, g, r) = self.image[x, y]
        print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

        # manipulate values and assign a new one
        self.image[x, y] = (0, 0, 255)
        (b, g, r) = self.image[x, y]
        print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

        return self.image


filepath = "/Users/pushpumkrishna/PycharmProjects/image_classification/sample_images/elephant.jpg"
newPath = "/Users/pushpumkrishna/PycharmProjects/image_classification/sample_images/elephant_new.jpg"
cv = OpenCV()
image_cv2 = cv.load_image_cv2(filepath=filepath)
image_matplot = cv.load_image_matplotlib()
# print(image_cv2)
# print(image_matplot)
cv.write_image(newPath, image_cv2)
img = cv.access_pixels(0, 0)
# print(img)
corner = img[0:100, 0:100]      # setting corner of image green
cv2.imshow("Corner", corner)
img[0:100, 0:100] = (0, 255, 0)
cv2.imshow("Updated", img)
cv2.waitKey(0)
