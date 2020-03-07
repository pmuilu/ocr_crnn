import os
import torch
import torchvision
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        files = os.listdir(self.root_dir)

        self.items = list(set([fname.split('.')[0] for fname in files]))
        self.transform = transform
        
        self.labels, self.char_vec = self._build_labels_and_char_vec()


    def _build_labels_and_char_vec(self):
        char_vec = set([])
        labels = []

        for item_name in self.items:
            with open(os.path.join(self.root_dir, item_name + ".gt.txt"), 'r') as gt:
                label = gt.read()
                labels.append(label)
                char_vec |= set(label)
        
        char_vec = np.sort(list(char_vec))
        char_vec = np.append(char_vec, "_") # BLANK character
        
        return np.array(labels), char_vec


    def __len__(self):
        return len(self.items)


    def get_num_classes(self):
        return len(self.char_vec)


    def get_encoded_label(self, index):
        return np.array([np.searchsorted(self.char_vec[:-1], 
                                         list(s)) for s in self.labels[index]])


    def get_decoded_label(self, label):
        return "".join(self.char_vec[label])


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_name = self.items[idx]
        
        image = Image.open(os.path.join(self.root_dir, item_name + ".bin.png"))

        if self.transform:
            image = self.transform(image)
        
        return image, self.get_encoded_label([idx])[0]

