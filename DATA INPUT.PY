import cv2
import os
import numpy as np
from sklearn import neighbors
import struct
import matplotlib.pyplot as plt
import gzip

def load_mnist(path, kind='train'):
   
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
 
file_path = "/home/zhang/Desktop/MNIST"

train_images, train_labels = load_mnist(file_path, "train")

test_images, test_labels = load_mnist(file_path, "t10k")

train_images.shape

    
    
