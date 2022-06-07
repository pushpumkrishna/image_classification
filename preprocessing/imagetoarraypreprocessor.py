# import the necessary packages
import tensorflow as tf
from datasets.dataset_loader import SimpleLoader
from preprocessing.dataset_processor import SimplePreprocessor


class ImageToArrayPreprocessor:

    def __init__(self,
                 dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of the image
        return tf.keras.utils.img_to_array(image, data_format=self.dataFormat)


sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleLoader(preprocessors=[sp, iap])
# (data, labels) = sdl.load_image(imagePaths, verbose=500)




