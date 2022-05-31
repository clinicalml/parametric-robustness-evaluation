from torch.utils.data import Dataset
import torch
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Subset
from torchvision import transforms
import os
from PIL import Image

class CelebADataset(Dataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """
    def __init__(self, img_path, meta_path, target_name, train_frac=0.7, val_frac=0.15):
        self.img_path = img_path
        self.meta_path = meta_path
        self.target_name = target_name


        # Load metadata
        self.attrs_df = pd.read_csv(os.path.join(self.meta_path, 'labels.csv'))
        
        # Store filenames separately
        self.file_names = self.attrs_df['file_path'].values
        self.attrs_df.drop("file_path", axis=1, inplace=True)

        # Get the y values
        self.y_array = torch.from_numpy(self.attrs_df[target_name].values).long()
        self.n_classes = 2

        # Get test / train splits
        train_size = int(np.round(float(len(self.y_array)) * train_frac))
        val_size = int(np.round(float(len(self.y_array)) * val_frac))
        test_size = len(self.y_array) - train_size - val_size
        self.split_array = np.concatenate((np.zeros(train_size), np.ones(val_size), 2*np.ones(test_size)))

        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.transform = get_transform()

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        y = self.y_array[idx]
        meta = self.attrs_df.loc[idx].values

        img_filename = os.path.join(
            self.img_path,
            self.file_names[idx])
        img = Image.open(img_filename).convert('RGB')
        x = self.transform(img)

        return x, y, meta

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            if train_frac<1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name


def get_transform():
    orig_w = 64
    orig_h = 64
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.CenterCrop(orig_min_dim),
        transforms.Resize(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform