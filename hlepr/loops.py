import sys
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from hlepr.util import AverageMeter, topk_accuracy
from hlepr.util_dynamic import realtime_classification, sample_weight_compute


def train_one_epoch(epoch, train_loader, model, criterion, optimizer, scaler, opt):
    'train the model in one epoch'
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        target = target.squeeze()
        for parameters in optimizer.param_groups:
            cur_lr = parameters["lr"]
        optimizer.zero_grad()
        if opt.amp:
            with autocast():
                output = model(input)
                loss = criterion(output, target)
                # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        acc1 = topk_accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.4f} ({top1.avg:.4f})'.format(epoch,
                                                               idx,
                                                               len(train_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               loss=losses,
                                                               top1=top1))
            sys.stdout.flush()  #

            print('* the train acc of this epoch is @{top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg, cur_lr


def train_one_epoch_dynamic(epoch, train_loader, model, criterion, optimizer, opt, sample_weight):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    sample_weight = sample_weight.cpu().detach().numpy()
    pred_list, target_list = [], []
    for idx, (input, target) in enumerate(train_loader):
        # regular traing
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        target = target.squeeze()
        output = model(input)
        loss = criterion(output, target)
        acc1 = topk_accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        for parameters in optimizer.param_groups:
            cur_lr = parameters["lr"]
        # adjust sample_weight
        if epoch % 100 == 0:
            pred_list, target_list = realtime_classification(output, target, pred_list, target_list)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.4f} ({top1.avg:.4f})'.format(epoch,
                                                               idx,
                                                               len(train_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               loss=losses,
                                                               top1=top1))
            print('* the train acc of this epoch is @{top1.avg:.3f}'.format(top1=top1))
        sys.stdout.flush()  #

    # after statistical classification results
    if pred_list:
        sample_weight = sample_weight_compute(pred_list, target_list, sample_weight)
        print("the adjusted sample_weight of this epoch ==>", sample_weight)
    else:
        sample_weight = torch.tensor(sample_weight, dtype=torch.float)

    return top1.avg, losses.avg, cur_lr, sample_weight

@torch.no_grad()
def validate(val_loader, model, criterion, opt):
    'for validation'

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            target = target.squeeze()
            output = model(input)
            loss = criterion(output, target)

            acc1 = topk_accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.4f} (top1.avg:.4f)'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

        print('the test acc of this epoch is {top1.avg:.4f}'.format(top1=top1))
    return top1.avg, losses.avg
