# Python libraries
from os import path
import random
import csv

# External modules
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
DATA_PATH = path.join('data', 'generated-data')
SEED = 69
BATCH_SIZE = 100
EPOCHS = 10

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
    train_X, train_y = train
    train_X, train_y = torch.from_numpy(train_X.values), torch.from_numpy(train_y.values)
    # train_X, train_y = train_X.type(torch.DoubleTensor), train_y.type(torch.LongTensor)
    val_X, val_y = val
    val_X, val_y = torch.from_numpy(val_X.values), torch.from_numpy(val_y.values)
    # val_X, val_y = val_X.type(torch.DoubleTensor), val_y.type(torch.LongTensor)

    simple_cnn = Net().to(device)
    simple_cnn = simple_cnn.float()
    
    optimizer = optim.Adam(simple_cnn.parameters(), lr = 0.001)
    loss_function = nn.CrossEntropyLoss() # multiclass, single label => categorical crossentropy as loss

    print(list(simple_cnn.parameters())[0].grad)

    for epoch in range(EPOCHS):
        batch_range = range(0, len(train_X), BATCH_SIZE) # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        for i in tqdm(batch_range): 
            # print(f"{i}:{i + BATCH_SIZE}")
            # print(i, i + BATCH_SIZE)
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 32, 32).float()
            # print(batch_X)
            batch_y = train_y[i:i + BATCH_SIZE].long()

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            simple_cnn.zero_grad()

            pred_y = simple_cnn(batch_X)
            # pred_y = func.softmax(pred_y, dim = 1)
            # Crossentropy berechnet intern schon den softmax https://discuss.pytorch.org/t/cross-entropy-loss-is-not-decreasing/43814/3
            loss = loss_function(pred_y, batch_y)
            loss.backward()
            optimizer.step() # Does the update

        print(batch_y)
        print(pred_y)

        print(f"Epoch: {epoch + 1}. Loss: {loss}")


def validate(test, cnn, device):
    test_X, test_y = test
    test_X, test_y = torch.from_numpy(test_X.values), torch.from_numpy(test_y.values)
    test_X, test_y = test_X.float(), test_y.long()

    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = cnn(test_X[i].view(-1, 1, 50, 50).to(device))[0]  # returns a list, 
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct / total, 3))


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 Inputchannel, 32 Filter somit Outputfeatures, 3 mal 3 Filtergröße
        self.conv1 = nn.Conv2d(1, 32, 3, stride = 1) 
        # 32 neue Inputchannel = trainierte Features als Input, 64 Filter darauf, 3 mal 3 px große Filter
        self.conv2 = nn.Conv2d(32, 64, 3) 
        # 64 Inputfeatures, 128 Outputfeatues, hier nun auf höherem Niveu, sodass ein feuerndes Neuron Anwesenheit einer
        self.conv3 = nn.Conv2d(64, 128, 3) 

        # First dense input = [batch_size, height * width * num_channels]
        # .view is the torch tensor version of numpy's reshape
        x = torch.randn(32, 32).view(-1, 1, 32, 32)
        self._conv_out_len = None
        self.conv_forward(x)

        self.dense1 = nn.Linear(self._conv_out_len, 512) # Number values of flattened tensor, ouput neurons
        self.dense2 = nn.Linear(512, 168) # 168 root letters to be predicted


    def conv_forward(self, x):
        # zuerst Aktivierung, auf den Aktivierungen dann erst größtes gewählt, 2 x 2 Pooling
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2)) 
        x = func.max_pool2d(func.relu(self.conv2(x)), (2, 2)) 
        x = func.max_pool2d(func.relu(self.conv3(x)), (2, 2))

        if self._conv_out_len is None:
            # x[0] because we need first element of the batch
            num_features = x[0].shape[0]
            num_px_height = x[0].shape[1]
            num_px_width = x[0].shape[2]
            self._conv_out_len = num_features * num_px_height * num_px_width

        return x

    
    def prepare_conv_to_dense(self, x):
        return x.view(-1, self._conv_out_len)
    

    def dense_forward(self, x):
        x = func.relu(self.dense1(x))
        x = func.relu(self.dense2(x))

        return x

    
    # def output_activation(self, x):
    #     # multiclass, single label => softmax
    #     return func.softmax(x, dim = 1)


    def forward(self, x):
        x = self.conv_forward(x)
        x = self.prepare_conv_to_dense(x)
        x = self.dense_forward(x)
        # x = self.output_activation(x)

        return x


def get_data(data_path: str, letter_part: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(path.join(data_path, '32by32-y-and-X.parquet'))
    data_y = data[letter_part] # data[data.columns[:3]]
    # data_y = pd.get_dummies(data_y[letter_part]), does not need to be one hot encoded for PyTorch cross entropy function
    data_X = data[data.columns[3:]]
    
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state = SEED)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = SEED)

    return train_X, train_y, val_X, val_y, test_X, test_y


def detect_gpu() -> None:
    device_num = torch.cuda.current_device()
    print(
        torch.cuda.device(device_num),
        torch.cuda.device_count(),
        torch.cuda.get_device_name(device_num),
        torch.cuda.is_available()
    )

class PerformanceTracker:
    
    def __init__(self):
        self.epoch_train_losses = []
        self.epoch_train_acc = []
        self.epoch_val_losses = []
        self.epoch_val_acc = []

    
    def add_train(self, loss, acc):
        self.epoch_train_losses.append(loss)
        self.epoch_train_acc.append(acc)

    
    def add_val(self, loss, acc):
        self.epoch_val_losses.append(loss)
        self.epoch_val_acc.append(acc)


    def save(self):
        save_frame = pd.DataFrame({
            'losses_train': self.epoch_train_losses,
            'accuracies_train': self.epoch_train_acc
            'losses_val': self.epoch_val_losses,
            'accuracies_val': self.epoch_val_acc
        })
        save_frame.to_csv(
            os.path.join('.', 'modelling', 'seperate_three', 'roots-epochs-metrics.csv'),
            quoting = csv.QUOTE_ALL
        )


if __name__ == '__main__':
    main()
