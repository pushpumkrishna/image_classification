from sklearn.datasets import fetch_openml

print("[INFO] accessing MNIST...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
print("[INFO] Shape of MNIST (full) dataset: ", X.shape)
# data = X
