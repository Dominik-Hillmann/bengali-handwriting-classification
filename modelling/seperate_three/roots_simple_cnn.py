# Python libraries
from os import path
import random
import csv
from statistics import mean
# External modules
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Internal modules
from utils.PerformanceTracker import PerformanceTracker
# Constants
DATA_PATH = path.join('data', 'generated-data')
SEED = 69
BATCH_SIZE = 100
EPOCHS = 50

# Settings
torch.manual_seed(SEED)
random.seed(SEED)

# Typing
from typing import Tuple



def main() -> None:
    detect_gpu()
    device = torch.device('cuda:0')
    print(device)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Running on GPU')
    else:
        device = torch.device('cpu')
        print('Running on CPU')

    train_X, train_y, val_X, val_y, test_X, test_y = get_data(DATA_PATH, 'grapheme_root')
    train((train_X, train_y), (val_X, val_y), device)
    # validate((test_X, test_y), device)
    

def train(
    train: Tuple[pd.DataFrame, pd.DataFrame], 
    val: Tuple[pd.DataFrame, pd.DataFrame],
    device: torch.device    
) -> None:

    tracker = PerformanceTracker(path.join('.', 'modelling', 'seperate_three'))

    train_X, train_y = train
    train_X, train_y = torch.from_numpy(train_X.values), torch.from_numpy(train_y.values)
    # train_X, train_y = train_X.type(torch.DoubleTensor), train_y.type(torch.LongTensor)
    val_X, val_y = val
    val_X, val_y = torch.from_numpy(val_X.values), torch.from_numpy(val_y.values)
    val_X, val_y = val_X.to(device), val_y.to(device)

    simple_cnn = ConvolutionalNeuralNet().to(device)
    simple_cnn = simple_cnn.float()
    
    optimizer = optim.Adam(simple_cnn.parameters(), lr = 0.001)
    loss_function = nn.CrossEntropyLoss() # multiclass, single label => categorical crossentropy as loss

    for epoch in range(EPOCHS):
        batch_range = range(0, len(train_X), BATCH_SIZE) # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        train_losses = []
        train_accuracies = []

        for i in tqdm(batch_range): 
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 32, 32).float()
            batch_y = train_y[i:i + BATCH_SIZE].long()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # pred_y = func.softmax(pred_y, dim = 1)
            # Crossentropy berechnet intern schon den softmax 
            # https://discuss.pytorch.org/t/cross-entropy-loss-is-not-decreasing/43814/3
            simple_cnn.zero_grad()
            pred_y = simple_cnn(batch_X)
            loss = loss_function(pred_y, batch_y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step() # Does the update

            pred_y_indices = torch.argmax(pred_y, dim = 1)
            num_correct = int((pred_y_indices == batch_y).int().sum())
            batch_size = int(batch_y.shape[0])
            accuracy = num_correct / batch_size
            train_accuracies.append(accuracy)

        val_loss, val_acc = validate((val_X, val_y), simple_cnn, loss_function, device)
        tracker.add_train(mean(train_losses), mean(train_accuracies))
        tracker.add_val(val_loss, val_acc)

        print(f'Epoch: {epoch + 1}')
        print(f'Train loss: {mean(train_losses)}')
        print(f'Train accuracy: {mean(train_accuracies)}')
        print(f'Val loss: {val_loss}')
        print(f'Val accuracy: {val_acc}')
    
    tracker.save('test-metrics.csv')
    tracker.graphs()


def validate(val, nn, loss_function, device):
    val_X, val_y = val
    # val_X, val_y = torch.from_numpy(val_X.values), torch.from_numpy(val_y.values)
    val_X, val_y = val_X.float(), val_y.long()
    val_X, val_y = val_X.to(device), val_y.to(device)

    correct = 0
    total = 0
    losses = []
    accuracies = []

    batch_range = range(0, len(val_X), BATCH_SIZE)
    with torch.no_grad():
        for i in batch_range:
            batch_X = val_X[i:i + BATCH_SIZE].view(-1, 1, 32, 32).float()
            batch_y = val_y[i:i + BATCH_SIZE].long()

            pred_y = nn(batch_X)
            loss = loss_function(pred_y, batch_y)
            losses.append(loss.item())

            pred_y_indices = torch.argmax(pred_y, dim = 1)
            num_correct = int((pred_y_indices == batch_y).int().sum())
            batch_size = int(batch_y.shape[0])
            accuracy = num_correct / batch_size
            accuracies.append(accuracy)

    return mean(losses), mean(accuracies)


class ConvolutionalNeuralNet(nn.Module):

    def __init__(
        self,
        *(num_1st_filter, num_2nd_filter, num_3rd_filter): Tuple[int, int, int],
        *(1st_layer_neurons, 2nd_layer_neurons): Tuple[int, int]
    ):
        super().__init__()
        # 32, 64, 128
        self.conv1 = nn.Conv2d(1, num_1st_filter, 3, stride = 1)
        self.conv2 = nn.Conv2d(num_1st_filter, num_2nd_filter, 3)
        self.conv3 = nn.Conv2d(num_2nd_filter, num_3rd_filter, 3) 

        # First dense input = [batch_size, height * width * num_channels]
        x = torch.randn(32, 32).view(-1, 1, 32, 32)
        self._conv_out_len = None
        self.conv_forward(x)

        self.dense1 = nn.Linear(self._conv_out_len, 512) # Number values of flattened tensor, ouput neurons
        self.dense2 = nn.Linear(512, 168) # 168 root letters to be predicted


    def conv_forward(self, x: torch.tensor) -> torch.tensor:

        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2)) 
        x = func.max_pool2d(func.relu(self.conv2(x)), (2, 2)) 
        x = func.max_pool2d(func.relu(self.conv3(x)), (2, 2))

        if self._conv_out_len is None:
            num_features = x[0].shape[0]
            num_px_height = x[0].shape[1]
            num_px_width = x[0].shape[2]
            self._conv_out_len = num_features * num_px_height * num_px_width

        return x

    
    def prepare_conv_to_dense(self, x: torch.tensor) -> torch.tensor:
        return x.view(-1, self._conv_out_len)
    

    def dense_forward(self, x: torch.tensor) -> torch.tensor:
        x = func.relu(self.dense1(x))
        x = func.relu(self.dense2(x))

        return x


    def forward(self, x):
        x = self.conv_forward(x)
        x = self.prepare_conv_to_dense(x)
        x = self.dense_forward(x)
        # x = self.output_activation(x)

        return x




def detect_gpu() -> None:
    device_num = torch.cuda.current_device()
    print(
        torch.cuda.device(device_num),
        torch.cuda.device_count(),
        torch.cuda.get_device_name(device_num),
        torch.cuda.is_available()
    )


if __name__ == '__main__':
    main()
