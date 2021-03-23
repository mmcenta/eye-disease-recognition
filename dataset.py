import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms


STOP_WORDS = set([
    "age", "the"
])


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


KEYWORD_MAP = {
    'Left-Diagnostic Keywords': 'left',
    'Right-Diagnostic Keywords': 'right',
}

def process_keywords(df):
    keywords = {} # indices for keywords for each eye
    word2idx, n_words = {'_': 0}, 1
    for col, key in KEYWORD_MAP.items():
        df[key] = df[col].str.replace(r'[^a-zA-Z\s]+', ' ', regex=True)
        df[key] = df[key].str.replace(r' +', ' ', regex=True)

        keywords[key] = []
        for idx in range(len(df)):
            item = df.iloc[idx]

            indices = []
            for kw in item[key].split():
                if kw in STOP_WORDS:
                    continue

                if kw not in word2idx:
                    word2idx[kw] = n_words
                    n_words += 1
                indices.append(word2idx[kw])

            keywords[key].append(torch.tensor(indices, dtype=torch.long))

    return word2idx, keywords


def collate_fn(batch):
    out = {'left_image': [], 'right_image': [], 'target': [], 'left_keyword': [], 'right_keyword': []}
    for item in batch:
        for key in out:
            out[key].append(item[key])
    for key in ('left_image', 'right_image', 'target'):
        out[key] = torch.stack(out[key])
    for key in ('left_keyword', 'right_keyword'):
        out[key] = pad_sequence(out[key])
    return out


class ODIRDataset(Dataset):
    def __init__(self, configs, csv_file, image_dir, **kwargs):
        super(ODIRDataset, self).__init__(**kwargs)

        self.csv_file = csv_file
        self.image_dir = image_dir
        self.image_size = configs['image_size']
        self.target_col = configs.get('target_col', -1)
        self.balanced = configs.get('balanced', False)
        self.h_flip = configs.get('h_flip', False)
        self.normalize = configs.get('normalize', False)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            get_augmentation_transform(self.h_flip, self.normalize),
        ])

        self._df = pd.read_csv(csv_file)
        self._df = remove_missing(self._df, image_dir)

        self.word2idx, self.keywords = process_keywords(self._df)
        self.idx2word = {i:w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
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
            'left_keyword': self.keywords['left'][idx],
            'right_keyword': self.keywords['right'][idx],
        }
