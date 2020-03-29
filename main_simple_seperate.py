# Python libraries
from os import path
import random

# External modules
import torch

# Internal modules
from utils.PerformanceTracker import PerformanceTracker
from utils.get_32px_data import get_32px_data
from utils.detect_gpu import detect_gpu, get_device
from modelling.seperate_three.ConvolutionalNeuralNet import ConvolutionalNeuralNet, train, validate

# Constants
DATA_PATH = path.join('data', 'generated-data')
SEED = 69
EPOCHS = 2
BATCH_SIZE = 100

# Settings
torch.manual_seed(SEED)
random.seed(SEED)

# Typing
from typing import Tuple


def main() -> None:
    detect_gpu()
    device = get_device()

    tracker = PerformanceTracker(path.join('.', 'modelling', 'seperate_three'))

    simple_cnn = ConvolutionalNeuralNet((32, 64, 128), (512, 168))
    train_X, train_y, val_X, val_y, test_X, test_y = get_32px_data(DATA_PATH, 'grapheme_root')
    train(simple_cnn, (train_X, train_y), (val_X, val_y), device, tracker, EPOCHS, BATCH_SIZE)


if __name__ == '__main__':
    main()
