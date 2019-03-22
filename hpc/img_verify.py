import os

import pandas as pd

from PIL import Image

with open(os.path.join(os.pardir, "train_image_path.txt"), 'r') as f:
    image_folder = f.readline().rstrip("\n")
    f.close()

data_folder = os.path.abspath(os.path.join(os.pardir, "data"))

train_df = pd.read_csv(os.path.join(data_folder, "processed", "train_image_df.csv"))
val_df = pd.read_csv(os.path.join(data_folder, "processed", "val_image_df.csv"))

invalid = []

print("1 / 2")

counter = 0
for path in train_df.image_path.values:
    if counter % 10000 == 0:
        print(counter / len(train_df))
    counter += 1
    try:
        img = Image.open(os.path.join(image_folder, path))
        img.verify()
        img.close()
    except:
        invalid.append(path)
        print("Corrupt or nonexisting file found")

print("2 / 2")

counter = 0
for path in val_df.image_path.values:
    if counter % 10000 == 0:
        print(counter / len(val_df))
    counter += 1
    try:
        img = Image.open(os.path.join(image_folder, path))
        img.verify()
        img.close()
    except:
        invalid.append(path)
        print("Corrupt or nonexisting file found")


corrupt_df = pd.DataFrame(data={"corrupt_path" : invalid})
corrupt_df.to_csv(os.path.join(data_folder, "external", "corrupt_files.csv"))