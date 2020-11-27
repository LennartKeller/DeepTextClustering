import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.texts)
