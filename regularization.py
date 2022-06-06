from typing import Tuple, Any
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
import argparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from datasets.dataset_loader import SimpleLoader
from preprocessing.dataset_processor import SimplePreprocessor

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# print(imagePaths)

sp = SimplePreprocessor(32, 32)
sdl = SimpleLoader(preprocessors=[sp])
(data, labels) = sdl.load_image(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.25,
                                                  random_state=5)

# loop over our set of regularizers
for r in (None, "l1", "l2"):
    # train an SGD classifier using a softmax loss function and the
    # specified regularization function for 10 epochs
    print("[INFO] training model with ‘{}‘ penalty".format(r))
    model = SGDClassifier(loss="log_loss",
                          penalty=r,
                          max_iter=100,
                          learning_rate="constant",
                          eta0=0.01,
                          random_state=42)
    model.fit(trainX, trainY)
    # evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] ‘{}‘ penalty accuracy: {:.2f}%".format(r, acc * 100))
