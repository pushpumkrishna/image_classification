# import the necessary packages
import cv2
import numpy as np
from imutils import paths
from keras.models import load_model
import argparse
from datasets.dataset_loader import SimpleLoader
from preprocessing.dataset_processor import SimplePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["cat", "dog", "panda"]

# grab the list of images that we’ll be describing
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
indexes = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[indexes]
# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load_image(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# make predictions on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image,
                "Label: {}".format(classLabels[preds[i]]),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.imshow("Image", image)
    cv2.waitKey(10)
    print("loop completed")


