import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def remove_missing(df, image_dir):
    missing = []
    for idx in  df.index:
        dd = df.iloc[idx]

        image_name = dd['Left-Fundus']
        image_path = os.path.join(image_dir, image_name)
        try:
            _ = Image.open(image_path)
        except FileNotFoundError:
            missing.append(idx)
            continue

        image_name = dd['Right-Fundus']
        image_path = os.path.join(image_dir, image_name)
        try:
            _ = Image.open(image_path)
        except FileNotFoundError:
            missing.append(idx)
        
    print("Removing {} training examples because the corresponding images were not found.".format(len(missing)))
    return df.drop(index=missing)


def load_image(path, transform):
    return transform(Image.open(path))


class ODIRDataset(Dataset):
    def __init__(self, csv_file, image_dir, img_size, **kwargs):
        super(ODIRDataset, self).__init__(**kwargs)

        self.csv_file = csv_file
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()])

        self._df = pd.read_csv(csv_file)
        self._df = remove_missing(self._df, image_dir)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        item = self._df.iloc[idx]

        # load images
        left_image = load_image(os.path.join(self.image_dir, item['Left-Fundus']), self.transform)
        right_image = load_image(os.path.join(self.image_dir, item['Right-Fundus']), self.transform)

        # load target
        target = list(map(int, item['target'][1:-1].split(',')))
        target = torch.argmax(torch.tensor(target, dtype=torch.long))

        return {
            'left_image': left_image,
            'right_image': right_image,
            'target': target,
        }
