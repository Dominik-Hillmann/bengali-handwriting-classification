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


def main() -> None:
    train_y_low_res = load_y()
    converted_batches = []

    for num_batch in range(4):
        print(f'Loading batch {num_batch + 1} of 4...')        
        train_X_batch = pd.read_parquet(os.path.join(DATA_PATH, f'train_image_data_{num_batch}.parquet'))

        collection = {}
        for i in tqdm(list(train_X_batch.index)):
            inter_batch_index = num_batch * BATCH_SIZE + i
            low_res_flat_pic = get_observation(train_X_batch, i)
            frame_index = f'Train_{inter_batch_index}'
            collection[frame_index] = low_res_flat_pic

        converted_batches.append(pd.DataFrame(
            list(collection.values()),
            index = list(collection.keys()),
            columns = PX_COLS
        ))

    save_converted_pics(train_y_low_res, converted_batches)
    


def load_y() -> pd.DataFrame:
    """Loads the file containing the labels.
    
    Returns:
        pd.DataFrame -- The `csv` file load into a dataframe.
    """

    train_y = pd.read_csv(
        os.path.join(DATA_PATH, 'train.csv'),
        index_col = 0
    )
    train_y_low_res = train_y[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
    # train_X_low_res = pd.DataFrame(columns = PX_COLS)

    return train_y_low_res#, train_X_low_res


def get_observation(train_X_batch: pd.DataFrame, i: int) -> list:
    """Get a formatted observation number `i` from this batch.
    It will a flattened and lower resolution version of the original picture.
    
    Arguments:
        train_X_batch {pd.DataFrame} -- The batch containing the original data.
        i {int} -- Observation number.
    
    Returns:
        list -- The flattened and downscaled observation.
    """

    high_res_pic = train_X_batch.iloc[i][1:].to_numpy(dtype = float)
    high_res_pic /= 255.0
    high_res_pic = 1.0 - high_res_pic
    high_res_pic = high_res_pic.reshape(ORIG_DIMS)
    low_res_pic = cv2.resize(high_res_pic, dsize = (SIZE, SIZE), interpolation = cv2.INTER_CUBIC)
    low_res_pic = low_res_pic.reshape((SIZE ** 2, ))

    return low_res_pic.tolist()


def save_converted_pics(train_y: pd.DataFrame, batches_list: list) -> None:
    """Save the converted data to a `.parquet` file.
    
    Arguments:
        train_y {pd.DataFrame} -- The labels of the observations
        batches_list {list} -- A list of the four converted batches.
    """

    train_X_low_res = pd.concat(batches_list, axis = 0, ignore_index = False)
    train_X_low_res = pd.concat([train_y, train_X_low_res], axis=1, sort=False)
    train_X_low_res.to_parquet(
        os.path.join(DATA_PATH, '32by32-y-and-X.parquet'),
        engine = 'pyarrow',
        index = True
    )


if __name__ == '__main__':
    main()
