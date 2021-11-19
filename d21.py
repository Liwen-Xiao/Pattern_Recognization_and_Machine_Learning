import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

def get_fashion_mnist_labels(labels):
    test_labels=['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [test_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d