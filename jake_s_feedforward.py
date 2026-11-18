'''A simple FeedForward Neural Network'''
import torch
from torch import nn

with open("/workspaces/Machine-Learning-Club-Practice/t10k-labels-idx1-ubyte", "rb") as f:
    labels = list(bytearray(f.read()))