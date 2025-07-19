"""Module for loading MNIST training and testing data"""

import os
import numpy as np

def load_images(filepath):
    """Load MNIST images from a UByte file,
    which is a file containing unsigned bytes.

    Each pixel value in MNIST images is stored as an unsigned 8-bit integer.
    This single byte represents a grayscale pixel ranging from 0 to 255.

    UByte files have a so-called "magic number", which is 4 bytes long.
    We won't be needing it so we'll skip those 4 bytes.
    These UByte files also have a bunch of metadata
    for the number of images, rows and columns.
    We'll use them to reshape the numpy array.
    """
    with open(filepath, 'rb') as f:
        f.read(4) # Skip magic number
        # Read metadata
        num_images = int.from_bytes(f.read(4))
        rows = int.from_bytes(f.read(4))
        cols = int.from_bytes(f.read(4))
        # Read images. Each byte in an unsigned int in the range 0.255
        images = np.frombuffer(f.read(), dtype=np.uint8)
        # `ìmages` is currently a flat vector. We have to reshape it
        images = images.reshape(num_images, rows * cols)
        return images

def load_labels(filepath):
    """Load MNIST labels from a UByte file.

    Each label is an integer from 0 to 9.
    
    Label files have less metadata than image files,
    since it only includes the number of labels.
    """
    with open(filepath, 'rb') as f:
        f.read(4) # Skip magic num
        f.read(4) # Skip metadata
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def load_data():
    """Load MNIST data from UByte files"""
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, '../../data')

    filenames = [
        'train-images.idx3-ubyte',
        'train-labels.idx1-ubyte',
        't10k-images.idx3-ubyte',
        't10k-labels.idx1-ubyte'
    ]
    filepaths = [os.path.join(data_dir, filename) for filename in filenames]

    x_train = load_images(filepaths[0])
    y_train = load_labels(filepaths[1])
    x_test = load_images(filepaths[2])
    y_test = load_labels(filepaths[3])
    return (x_train, y_train), (x_test, y_test)
