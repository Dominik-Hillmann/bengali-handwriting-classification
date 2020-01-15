import pandas as pd
import os
test = pd.read_parquet(os.path.join('.', 'data', 'bengaliai-cv19', 'train_image_data_0.parquet'))
print(test.head(10))
print(test.shape)
print(test.columns)
