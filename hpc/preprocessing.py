import os

with f as open(os.path.join(os.pardir, "train_path.txt")):
    train_folder = f.readline()
    f.close()

with f as open(os.path.join(os.pardir, "train_image_path.txt")):
    train_image_folder = f.readline()
    f.close()

print(train_folder, train_image_folder)

