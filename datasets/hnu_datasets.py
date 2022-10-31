import torch
from torch.utils.data import Dataset
from datasets.preprocess import max_min_function, load_hnu_data


class HNU_Datasets(Dataset):
    def __init__(self, work_dir, size, step, length):
        self.data_info = load_hnu_data(work_dir, size, step, length, preprocess_function=max_min_function)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, item):
        vibration, label = self.data_info[item]
        vibration = torch.tensor(vibration, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return vibration, label
