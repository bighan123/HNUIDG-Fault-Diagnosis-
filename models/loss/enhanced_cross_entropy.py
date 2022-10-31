import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedCrossEntropyLoss(nn.Module):
    def __init__(self, size, num_cls):
        super().__init__()
        if isinstance(size, int):
            weight = [1 / num_cls] * num_cls
        elif isinstance(size, list):
            size = sum(size) / size
            weight = size / sum(size)
        self.weight = torch.tensor(weight, dtype=torch.float)

    def forward(self, output, target):
        output = torch.softmax(output, dim=-1)
        compensation_factor = torch.exp(1 - output).mean(1)
        cross_entropy_loss = F.cross_entropy(output, target, weight=self.weight)
        loss = (compensation_factor * cross_entropy_loss).mean()
        return loss
