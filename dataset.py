import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
import pandas as pd

class ECGDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file, delim_whitespace=True)  # 使用空格作为分隔符
        print("Data frame shape:", self.data_frame.shape)  # 添加调试信息
        self.labels = self.data_frame.iloc[:, 0].values
        self.data = self.data_frame.iloc[:, 1:].values

        print("Number of samples:", len(self.data))  # 添加调试信息

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index], dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        label = self.labels[index]
        return sample, label