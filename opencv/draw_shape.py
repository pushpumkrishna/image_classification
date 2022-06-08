import numpy as np
import cv2


class AlterImage:

    def __init__(self):
        pass

    def generate_image(self,
                       width: int,
                       height: int):
        """

        :return:
        """
        self.width = width
        self.height = height
        self.image = np.zeros((self.width, self.height, 3), dtype="uint8")

    def draw_line(self):
        """

        :return:
        """
        green = (0, 255, 0)
        cv2.line(img=self.image,
                 pt1=(0, 0),
                 pt2=(300, 300),
                 color=green)

        cv2.imshow("Canvas", self.image)
        cv2.waitKey(0)

    def draw_rectangle(self):
        """

        :return:
        """
        red = (0, 0, 255)  # BGR
        cv2.rectangle(img=self.image,
                      pt1=(10, 10),
                      pt2=(160, 160),  # rectangle size = 150 x 150 pixels
                      color=red,
                      thickness=5)  # set -1 if you want a solid colour rectangle

        cv2.imshow("Canvas", self.image)
        cv2.waitKey(0)

    def draw_circle(self):
        """

        :return:
        """
        red = (0, 0, 255)  # BGR

        (centerX, centerY) = (self.image.shape[1] // 2, self.image.shape[0] // 2)
        white = (255, 255, 255)

        for r in range(0, 175, 25):
            cv2.circle(self.image, (centerX, centerY), r, white)

        cv2.imshow("Canvas", self.image)
        cv2.waitKey(0)

    def multiple_cicles(self):
        """

        :return:
        """

        for i in range(0, 25):
            radius = np.random.randint(0, high=100)
            colour = np.random.randint(0, high=256, size=(3,)).tolist()
            points = np.random.randint(0, high=300, size=(2,))

            cv2.circle(img=self.image,
                       center=tuple(points),
                       radius=radius,
                       color=colour,
                       thickness=-1)

        cv2.imshow("Image", self.image)
        cv2.waitKey(0)


# al = AlterImage()
# al.generate_image(300, 300)
# al.draw_line()
# al.draw_rectangle()
# al.draw_circle()
# al.multiple_cicles()
