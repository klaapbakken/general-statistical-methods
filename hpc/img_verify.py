import os

import pandas as pd

from PIL import Image

with open(os.path.join(os.pardir, "train_image_path.txt"), 'r') as f:
    image_folder = f.readline().rstrip("\n")
    f.close()

data_folder = os.path.abspath(os.path.join(os.pardir, "data"))

train_df = pd.read_csv(os.path.join(data_folder, "processed", "train_image_df.csv"))

invalid = []
for path in train_df.image_path.values:
    try:
        img = Image.open(os.path.join(image_folder, path))
        img.verify()
        img.close()
    except:
        invalid.append(path)
        print("Corrupt or nonexisting file founds")