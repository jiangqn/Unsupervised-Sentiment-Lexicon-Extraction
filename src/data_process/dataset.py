import torch
from torch.utils.data.dataset import Dataset
import numpy as np

class ClassifierDataset(Dataset):

    def __init__(self, path):
        data = np.load(path)
        self.sentence = torch.from_numpy(data['sentence']).long()
        self.label = torch.from_numpy(data['label']).long()
        self.len = self.label.size(0)

    def __getitem__(self, item):
        return self.sentence[item], self.label[item]

    def __len__(self):
        return self.len