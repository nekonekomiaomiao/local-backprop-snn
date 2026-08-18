import gzip
import os
import sys
import numpy as np

# PyInstaller frozen single-file mode: data bundled into _MEIPASS
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _base = str(sys._MEIPASS)
else:
    _base = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_base, "mnist_data")


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic == 2051:
            n, rows, cols = np.frombuffer(f.read(12), dtype=">u4", count=3)
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(int(n), int(rows), int(cols))
        elif magic == 2049:
            n = int.from_bytes(f.read(4), "big")
            data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        else:
            raise ValueError(f"bad magic {magic}")
        return data


def load_mnist(train=True):
    if train:
        imgs = _read_idx(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
        lbls = _read_idx(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
    else:
        imgs = _read_idx(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
        lbls = _read_idx(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))
    return imgs, lbls


if __name__ == "__main__":
    tr, trl = load_mnist(train=True)
    te, tel = load_mnist(train=False)
    print("train:", tr.shape, trl.shape, "unique labels", np.unique(trl))
    print("test :", te.shape, tel.shape)
    print("pixel range", tr.min(), tr.max(), "mean", tr.mean().round(2))
