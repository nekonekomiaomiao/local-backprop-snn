import os
from mnist_loader import _read_idx

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_fashion_data")


def load_mnist(train=True):
    if train:
        imgs = _read_idx(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
        lbls = _read_idx(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
    else:
        imgs = _read_idx(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
        lbls = _read_idx(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))
    return imgs, lbls
