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


def get_augmentation_transform(h_flip=False, normalize=False):
    aug_list = []
    if h_flip:
        aug_list.append(transforms.RandomHorizontalFlip())
    aug_list.extend([
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1),
        transforms.RandomGrayscale(),
    ])
    if normalize:
        aug_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(aug_list)


class ODIRDataset(Dataset):
    def __init__(self, csv_file, image_dir, img_size, target_col, balanced=False, normalize=False, h_flip=False, **kwargs):
        super(ODIRDataset, self).__init__(**kwargs)

        self.csv_file = csv_file
        self.image_dir = image_dir
        self.image_size = img_size
        self.target_col = target_col
        self.balanced = balanced
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            get_augmentation_transform(h_flip, normalize),
        ])

        self._df = pd.read_csv(csv_file)
        self._df = remove_missing(self._df, image_dir)
        
        self.targets, self.weight = [], torch.zeros(8) if self.target_col < 0 else torch.zeros(2)
        for idx in range(len(self._df)):
            item = self._df.iloc[idx]
            
            target = torch.tensor(list(map(int, item['target'][1:-1].split(','))), dtype=torch.long)
            if self.target_col < 0:
                target = torch.argmax(target)
            else:
                target = (torch.argmax(target) == self.target_col).long()

            self.targets.append(target)
            self.weight[target] += 1
        
        self.weight  = self.weight / torch.sum(self.weight)

    def get_weight(self):
        if self.weight.size(0) > 2:
            return self.weight
        return self.weight[1]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        item = self._df.iloc[idx]

        # load images
        left_image = load_image(os.path.join(self.image_dir, item['Left-Fundus']), self.transform)
        right_image = load_image(os.path.join(self.image_dir, item['Right-Fundus']), self.transform)

        return {
            'left_image': left_image,
            'right_image': right_image,
            'target': self.targets[idx],
        }
