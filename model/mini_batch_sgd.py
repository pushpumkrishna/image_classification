from typing import Tuple, Any
import numpy as np
from numpy import exp, arange
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import argparse


def sigmoid(x):

    return 1.0 / (1 + exp(-x))


def predict(X, W):

    prediction = sigmoid(X.dot(W))

    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0

    return prediction


def next_batch(X, y, batch_size) -> Tuple[Any, Any]:
    """

    :return:type batch_size: tuple
    """
    for i in arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="number of epochs")
ap.add_argument("-l", "--learning", type=float, default=0.001, help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of input batch")
args = vars(ap.parse_args())


# generate a 2-class classification problem with 1,000 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
# insert a column of 1’s as the last entry in the feature
# matrix -- this little trick allows us to treat the bias
# as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of
# the data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.25, random_state=42)


# initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in arange(0, args["epochs"]):

    epochLoss = []

    for (batchX, batchY) in next_batch(trainX, trainY, args["batch_size"]):

        # take the dot product between our current batch of features
        # and the weight matrix, then pass this value through our
        # activation function
        prediction = sigmoid(batchX.dot(W))

        # now that we have our predictions, we need to determine the
        # ‘error‘, which is the difference between our predictions
        # and the true values
        error = batchY - prediction
        epochLoss.append(np.sum(error ** 2))

        # the gradient descent update is the dot product between our
        # current batch and the error on the batch

        gradient = batchX.T.dot(error)

        # in the update stage, all we need to do is "nudge" the
        # weight matrix in the negative direction of the gradient
        # (hence the term "gradient descent") by taking a small step
        # towards a set of "more optimal" parameters
        W += -args["learning"] * gradient

    # update our loss history by taking the average loss across all batches
    loss = np.average(epochLoss)
    losses.append(loss)

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))


# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))
