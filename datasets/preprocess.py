import numpy as np
import os
import pandas as pd
import scipy.io as scio


def max_min_function(data):
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    data_norm = 2 * (data - data_min) / (data_max - data_min) - 1
    return data_norm


def load_hnu_data(work_dir, size, step, length, preprocess_function, use_preprocess=True):
    # loading bearing datasets from hnu
    data_info = list()
    label = 0
    if not os.path.exists(work_dir):
        raise FileExistsError('the work_dir is not found')
    for _, _, file in os.walk(work_dir):
        for each_file in file:
            file_name = os.path.join(work_dir, each_file)
            data = pd.read_excel(file_name)
            data = data.values
            data_x = np.array(data[..., 0])
            data_y = np.array(data[..., 1])
            data_z = np.array(data[..., 2])
            if use_preprocess:
                data_x = preprocess_function(data_x)
                data_y = preprocess_function(data_y)
                data_z = preprocess_function(data_z)
            if isinstance(size, int):
                for i in range(size):
                    each_data_x = data_x[None, i * step:i * step + length]
                    each_data_y = data_y[None, i * step:i * step + length]
                    each_data_z = data_z[None, i * step:i * step + length]
                    each_data = np.concatenate([each_data_x, each_data_y, each_data_z], axis=0)
                    data_info.append((each_data, np.array([label])))
            elif isinstance(size, list):
                for i in range(size[label]):
                    each_data_x = data_x[None, i * step:i * step + length]
                    each_data_y = data_y[None, i * step:i * step + length]
                    each_data_z = data_z[None, i * step:i * step + length]
                    each_data = np.concatenate([each_data_x, each_data_y, each_data_z], axis=0)
                    data_info.append((each_data, np.array([label])))
            label += 1
    return data_info


def convert_data_type(data):
    for i in range(len(data)):
        if type(data[i]) == str:
            data[i] = float(data[i])
    return data


def load_xjtu_data(work_dir, size, step, length, preprocess_function, use_preprocess=True):
    # loading bearing datasets from XJTU
    label = 0
    data_info = list()
    for dir_name, _, _ in os.walk(work_dir):
        if dir_name != work_dir:
            for _, _, file in os.walk(dir_name):
                concat_channel = []
                for each_file in file:
                    channel = []
                    file_name = os.path.join(dir_name, each_file)
                    data = pd.read_table(file_name, low_memory=False)
                    data = data.values
                    # 读取稳定运行后的数据,并转为浮点数
                    data = convert_data_type(np.array(data[..., 0][1000:]))
                    if use_preprocess:
                        data = preprocess_function(data)
                    if isinstance(size, int):
                        for i in range(size):
                            data_one_channel = data[None, i * step: i * step + length]
                            channel.append(data_one_channel)
                    elif isinstance(size, list):
                        for i in range(size[label]):
                            data_one_channel = data[None, i * step: i * step + length]
                            channel.append(data_one_channel)
                    concat_channel.append(channel)
                for i in range(len(concat_channel[0])):
                    data_mixed = np.concatenate([concat_channel[0][i], concat_channel[1][i]], axis=0)
                    data_info.append((data_mixed, np.array([label])))
            label += 1
    return data_info


def load_dds_data(work_dir, size, step, length, preprocess_function, use_preprocess=True):
    label = 0
    data_info = list()
    for dir_name, _, _ in os.walk(work_dir):
        if dir_name != work_dir:
            for _, _, file in os.walk(dir_name):
                for each_file in file:
                    file_name = os.path.join(dir_name, each_file)
                    data = pd.read_table(file_name, low_memory=False)
                    data_x = data['Unnamed: 3'].values
                    data_y = data['Unnamed: 4'].values
                    data_z = data['Unnamed: 5'].values
                    # 转换类型,读取稳定运行后的数据
                    data_x = np.array([float(v) for v in data_x[100:]])
                    data_y = np.array([float(v) for v in data_y[100:]])
                    data_z = np.array([float(v) for v in data_z[100:]])
                    if use_preprocess:
                        data_x = preprocess_function(data_x)
                        data_y = preprocess_function(data_y)
                        data_z = preprocess_function(data_z)
                    if isinstance(size, int):
                        for i in range(size):
                            each_data_x = data_x[None, i * step:i * step + length]
                            each_data_y = data_y[None, i * step:i * step + length]
                            each_data_z = data_z[None, i * step:i * step + length]
                            each_data = np.concatenate([each_data_x, each_data_y, each_data_z], axis=0)
                            data_info.append((each_data, np.array([label])))
                    elif isinstance(size, list):
                        for i in range(size[label]):
                            each_data_x = data_x[None, i * step:i * step + length]
                            each_data_y = data_y[None, i * step:i * step + length]
                            each_data_z = data_z[None, i * step:i * step + length]
                            each_data = np.concatenate([each_data_x, each_data_y, each_data_z], axis=0)
                            data_info.append((each_data, np.array([label])))
            label += 1
    return data_info
