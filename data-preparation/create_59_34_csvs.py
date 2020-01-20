# Python libraries
import os
import sys
# External libraries
import pandas as pd
import cv2


DATA_PATH = os.path.join('data', 'bengaliai-cv19')
SIZE = 32

def main():
    train_targets = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    train_data = []
    lower_res_data = train_targets[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]

    for num_batch in range(4):
        train_data = pd.read_parquet(os.path.join(DATA_PATH, f'train_image_data_{num_batch}.parquet'))

        for i in list(train_data.index):
            high_res_pic = train_data.iloc[i][1:].to_numpy(dtype = float)
            high_res_pic /= 255.0
            high_res_pic = 1.0 - high_res_pic
            low_res_pic = cv2.resize(high_res_pic, dsize = (SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
            low_res_pic = low_res_pic.reshape((SIZE ** 2, ))

    # Target: Make new parquet file with y and X in lower resolution

if __name__ == '__main__':
    main()


