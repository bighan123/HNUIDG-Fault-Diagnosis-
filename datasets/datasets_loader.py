import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler



def get_loaders(work_dir, size, step, length, train_ratio, Mydataset, batch_size=32, num_workers=0):
    vibration_dataset = Mydataset(work_dir, size, step, length)
    total_size = len(vibration_dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(vibration_dataset,
                                                                lengths=[train_size, test_size],
                                                                generator=torch.manual_seed(0))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)
    return train_loader, test_loader


def get_loaders_new(opt, MyDatasets):
    train_index, test_index = [], []
    if opt.use_ratio:
        # 默认在样本平衡条件下训练
        vibration_dataset = MyDatasets(opt.work_dir,opt.size,opt.step,opt.length)
        size = [opt.size] * opt.num_cls
        for i in range(len(size)):
            shuffle_idx = np.random.permutation(size[i])
            if i == 0:
                each_train_idx = shuffle_idx[:int(size[i] * opt.ratio)]
                each_test_idx = shuffle_idx[int(size[i] * opt.ratio):]
            else:
                each_train_idx = shuffle_idx[:int(size[i] * opt.ratio)] + sum(size[:i])
                each_test_idx = shuffle_idx[int(size[i] * opt.ratio):] + sum(size[:i])
            train_index.extend(each_train_idx)
            test_index.extend(each_test_idx)
    else:
        # 指定训练和测试的样本数
        train_size = opt.train_size
        test_size = opt.test_size
        num_cls = opt.num_cls
        if isinstance(train_size, int):
            train_size = [train_size] * num_cls
        elif isinstance(train_size, list):
            if len(train_size) == 2:
                train_size = [train_size[0]] + [train_size[1]] * (num_cls - 1)
            else:
                assert len(train_size) == num_cls, 'if you want to use imbalanced sample to train, You must specify a number of training samples for each category '
        total_size =[train_size[i] + test_size for i in range(len(train_size))]
        vibration_dataset = MyDatasets(opt.work_dir, total_size, opt.step, opt.length)
        for i in range(len(total_size)):
            shuffle_idx = np.random.permutation(total_size[i])
            if i == 0:
                each_train_idx = shuffle_idx[:train_size[i]]
                each_test_idx = shuffle_idx[train_size[i]:]
            else:
                each_train_idx = shuffle_idx[:train_size[i]] + sum(total_size[:i])
                each_test_idx = shuffle_idx[train_size[i]:] + sum(total_size[:i])
            train_index.extend(each_train_idx)
            test_index.extend(each_test_idx)
        random.shuffle(train_index)
        random.shuffle(test_index)


    train_loader = DataLoader(dataset=vibration_dataset,
                              batch_size=opt.batch_size,
                              num_workers=opt.num_workers,
                              sampler=SubsetRandomSampler(train_index),
                              pin_memory=True)
    test_loader = DataLoader(dataset=vibration_dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.num_workers,
                             sampler=SubsetRandomSampler(test_index),
                             pin_memory=True)
    return train_loader, test_loader
