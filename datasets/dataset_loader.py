import cv2
import os
from numpy import array


class SimpleLoader:

    def __init__(self,
                 preprocessors=None):

        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load_image(self,
                   imagepaths,
                   verbose=-1
                   ):

        data = []
        labels = []

        for (i, imagePath) in enumerate(imagepaths):
            # load the image and extract the class label
            # path format:
            # ./dataset/{class}/{image}.jpg

            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:

                for p in self.preprocessors:
                    image = p.preprocess(image)

                data.append(image)
                labels.append(label)

                # show an update every ‘verbose‘ images
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1,
                                                          len(imagepaths)))

        # noinspection PyRedundantParentheses
        return (array(data), array(labels))
