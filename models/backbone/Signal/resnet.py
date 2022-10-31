import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    '''
    The BasicBlock for building resnet18 and resnet50
    '''
    expansion = 1

    def __init__(self, in_c, out_channels, stride=1,
                 is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv1d(in_channels=in_c,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels * BasicBlock.expansion,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels * BasicBlock.expansion)
        if stride != 1 or in_c != out_channels * BasicBlock.expansion:
            self.short_cut = nn.Sequential(nn.Conv1d(in_channels=in_c,
                                                     out_channels=out_channels * BasicBlock.expansion,
                                                     kernel_size=1,
                                                     stride=stride,
                                                     bias=False),
                                           nn.BatchNorm1d(out_channels * BasicBlock.expansion))
        else:
            self.short_cut = nn.Identity()

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        out = x + self.short_cut(short_cut)
        return F.relu(out, True)
        # 残差连接


class BottleNeck(nn.Module):
    '''
    The BasicBlock for building resnet over 50 years
    '''

    expansion = 4

    def __init__(self, in_c, out_channels, stride=1,
                 is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv1d(in_channels=in_c,
                               out_channels=out_channels,
                               stride=1,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.act1 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(True)

        self.conv3 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels * BottleNeck.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * BottleNeck.expansion)
        if stride != 1 or in_c != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels * BottleNeck.expansion)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        out = x + self.shortcut(short_cut)
        return F.relu(out, True)


class ResNet(nn.Module):
    def __init__(self, block, num_block, h_args, in_c, num_cls=8, use_init=True):
        super().__init__()

        self.in_channels = 64
        self.in_c = in_c
        self.num_cls = num_cls
        self.conv1 = nn.Conv1d(in_channels=in_c,
                               out_channels=64,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU(True)

        self.stage1 = self._make_layers(block, 64, num_block[0], 1)
        self.stage2 = self._make_layers(block, 128, num_block[1], 2)
        self.stage3 = self._make_layers(block, 256, num_block[2], 2)
        self.stage4 = self._make_layers(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_channels = 512 * block.expansion
        self.classifier = nn.ModuleList()
        if h_args is None:
            self.classifier.append(nn.Linear(self.last_channels, num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        else:
            for i in range(len(h_args)):
                if i == 0:
                    self.classifier.append(nn.Linear(self.last_channels, h_args[i]))
                else:
                    self.classifier.append(nn.Linear(h_args[i - 1], h_args[i]))
            self.classifier.append(nn.Linear(h_args[-1], num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        if use_init:
            self._initialize_weights()

    def forward(self, x, is_last=False):
        b = x.shape[0]
        x = self.conv1(x)
        f0 = x
        x = self.stage1(x)
        f1 = x
        x = self.stage2(x)
        f2 = x
        x = self.stage3(x)
        f3 = x
        x = self.stage4(x)
        f4 = x
        x = self.avg_pool(x)
        f5 = x
        x = x.view(b, -1)
        for module in self.classifier:
            x = module(x)
        if is_last:
            return [f0, f1, f2, f3, f4, f5], x
        else:
            return x

    def _make_layers(self, block, out_channels, num_blocks, stride):
        '''

        :param block: the type of block, BasicBlock or BottleNeck
        :param out_channels: output depth channel nummer of this layer
        :param num_blocks: hao many blocks using this year
        :param stride: the stride of the first block in this year
        :return: resnet layer
        '''
        # in each stage, the stride of first block should be the specific stirde
        # and the stride of the other block should be one
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers += [block(self.in_channels, out_channels, s)]
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def get_feat_module(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.stage1)
        feat_m.append(self.stage2)
        feat_m.append(self.stage3)
        feat_m.append(self.stage4)
        return feat_m

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def resnet18(h_args, in_c, num_cls):
    model = ResNet(BasicBlock, num_block=[2, 2, 2, 2],
                   h_args=h_args, in_c=in_c, num_cls=num_cls)
    return model


def resnet34(h_args, in_c, num_cls):
    model = ResNet(BasicBlock, num_block=[3, 4, 6, 3],
                   h_args=h_args, in_c=in_c, num_cls=num_cls)
    return model


def resnet50(h_args, in_c, num_cls):
    model = ResNet(BottleNeck, num_block=[3, 4, 6, 3],
                   h_args=h_args, in_c=in_c, num_cls=num_cls)
    return model


def resnet101(h_args, in_c, num_cls):
    model = ResNet(BottleNeck, num_block=[3, 4, 23, 3],
                   h_args=h_args, in_c=in_c, num_cls=num_cls)
    return model


def resnet152(h_args, in_c, num_cls):
    model = ResNet(BottleNeck, num_block=[3, 8, 36, 3],
                   h_args=h_args, in_c=in_c, num_cls=num_cls)
    return model
