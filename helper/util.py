import torch
import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        b = target.size(0)
        values, preds = torch.topk(input=output,
                                   k=maxk,
                                   dim=1,
                                   largest=True,
                                   sorted=True)
        preds = preds.t()
        true_match = preds.eq(target.view(1, -1).expand_as(preds))

        ans = []
        for k in topk:
            true_match = true_match[:k].view(-1).float().sum(0, keepdim=True)
            k_accuracy = torch.mul(true_match, 100 / b)
            ans.append(k_accuracy)
        return ans


def adjust_lr(epoch, opt, optimizer):
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
