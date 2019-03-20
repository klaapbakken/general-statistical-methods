import os

with open(os.path.join(os.pardir, "train_path.txt"), 'r') as f:
    train_folder = f.readline().rstrip("\n")
    f.close()

with open(os.path.join(os.pardir, "train_image_path.txt"), 'r') as f:
    train_image_folder = f.readline().rstrip("\n")
    f.close()

print(train_folder, train_image_folder)

