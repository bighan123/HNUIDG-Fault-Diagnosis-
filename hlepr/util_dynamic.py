import torch
import numpy as np

def realtime_classification(output, target, pred_list, target_list):
    _, pred = torch.max(output, dim=-1)
    pred_list.extend(pred.cpu().detach().numpy())
    target_list.extend(target.cpu().detach().numpy())
    return pred_list, target_list


def sample_weight_compute(pred_list, target_list, sample_weight):
    label_match = [0] * len(np.unique(target_list))
    target_num = [0] * len(np.unique(target_list))
    for each_target in target_list:
        target_num[each_target] += 1
    for i in range(len(target_list)):
        label_match[target_list[i]] += 1 if pred_list[i] == target_list[i] else 0
    miss_match = [target_num[i] - label_match[i] for i in range(len(target_num))]
    for i in range(len(miss_match)):
        if miss_match[i] != 0:
            sample_weight[i] = sample_weight[i] * miss_match[i]
    sample_weight = torch.tensor(sample_weight, dtype=torch.float)
    return (sample_weight / sum(sample_weight))
