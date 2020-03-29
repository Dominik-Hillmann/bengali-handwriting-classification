# Python libraries
from os import path

# External modules
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
SEED = 69

# Typing
from typing import Tuple


def get_32px_data(
    data_path: str, 
    letter_part: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    data = pd.read_parquet(path.join(data_path, '32by32-y-and-X.parquet'))
    data_y = data[letter_part]
    # data_y = pd.get_dummies(data_y[letter_part]) Does not need to be one hot encoded for PyTorch cross entropy function
    data_X = data[data.columns[3:]]
    
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state = SEED)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size = 0.2, random_state = SEED)

    return train_X, train_y, val_X, val_y, test_X, test_y
