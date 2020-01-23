# Python libraries
import os

# External libraries
import pandas as pd
import cv2
from tqdm import tqdm

# Important constants
DATA_PATH = os.path.join('data', 'bengaliai-cv19')
SIZE = 32
PX_COLS = [f'px_{i}' for i in range(SIZE ** 2)]
ORIG_DIMS = (137, 236)
BATCH_SIZE = 50210


def main():
    train_y_low_res, train_X_low_res = load_meta_data()
    # px_lists = [[]] * (SIZE ** 2)
    collection = {}
    print(train_y_low_res.head(5))
    print(train_X_low_res.head(5))

    for num_batch in range(1):
        print(f'Loading batch {num_batch + 1} of 4...')        
        train_X_batch = pd.read_parquet(os.path.join(DATA_PATH, f'train_image_data_{num_batch}.parquet'))

        for i in tqdm(list(train_X_batch.index)):
            inter_batch_index = num_batch * BATCH_SIZE + i
            low_res_flat_pic = get_observation(train_X_batch, i)
            frame_index = f'Train_{inter_batch_index}'
            collection[frame_index] = low_res_flat_pic
            # train_X_low_res.loc[frame_index] = low_res_flat_pic
            # ineffizient, jedes einzelne Element anhängen wird langsamer je größer der DataFrame
            # => Listen in Dict, mit key


        # print(train_X_low_res.head(10))
        print(pd.DataFrame(list(collection.values()), index = list(collection.keys())))


def load_meta_data():
    train_y = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    train_y_low_res = train_y[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
    train_X_low_res = pd.DataFrame(columns = PX_COLS)

    return train_y_low_res, train_X_low_res


def get_observation(train_X_batch, i):
    high_res_pic = train_X_batch.iloc[i][1:].to_numpy(dtype = float)
    high_res_pic /= 255.0
    high_res_pic = 1.0 - high_res_pic
    high_res_pic = high_res_pic.reshape(ORIG_DIMS)
    low_res_pic = cv2.resize(high_res_pic, dsize = (SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
    low_res_pic = low_res_pic.reshape((SIZE ** 2, ))

    return low_res_pic.tolist()


def append_to_px_lists(flat_low_res_img, px_lists):
    flat_low_res_img = list(flat_low_res_img)
    for i in range(len(flat_low_res_img)):
        px_lists[i].append(flat_low_res_img[i])

    return px_lists


def add_obs_low_res_X(low_res_X):
    raise NotImplementedError


if __name__ == '__main__':
    main()
