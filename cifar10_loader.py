#!/usr/bin/env python3
"""CIFAR-10 loader：32x32x3 -> 灰度 -> 中央 28x28 裁剪（与 MNIST 同输入规格，复用主拓扑 784）。
数据：cifar10_data/cifar-10-batches-py/（官方 cifar-10-python.tar.gz 解压）。"""
import os
import pickle
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar10_data", "cifar-10-batches-py")


def _unpickle(f):
    with open(f, "rb") as fh:
        return pickle.load(fh, encoding="bytes")


def load_mnist(train=True):
    # 命名与 mnist_loader.load_mnist 对齐
    if train:
        imgs, lbls = [], []
        for i in range(1, 6):
            d = _unpickle(os.path.join(D, f"data_batch_{i}"))
            imgs.append(d[b"data"])
            lbls.extend(d[b"labels"])
        imgs = np.concatenate(imgs)
        lbls = np.array(lbls)
    else:
        d = _unpickle(os.path.join(D, "test_batch"))
        imgs = d[b"data"]
        lbls = np.array(d[b"labels"])
    imgs = imgs.reshape(-1, 3, 32, 32)
    gray = (0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]).astype(np.uint8)
    crop = gray[:, 2:30, 2:30]          # 中央 28x28
    return crop, lbls
