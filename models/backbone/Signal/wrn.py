import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_ratio=0., use_bias=False):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=use_bias)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=use_bias)
        self.drop_ratio = drop_ratio
        self.equal = in_channels == out_channels
        self.stride = stride
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=use_bias)

    def forward(self, x):
        x = self.bn1(x)
        x = self.act1(x)
        short_cut = x
        out = self.act2(self.bn2(self.conv1(x)))
        if self.drop_ratio > 0:
            out = F.dropout(out, p=self.drop_ratio, training=self.training)
        out = self.conv2(out)
        if not self.equal or self.stride > 1:
            short_cut = self.shortcut(short_cut)
        return torch.add(short_cut, out)


class WrnBlock(nn.Module):
    def __init__(self, stack_layers, in_channels, out_channels,
                 basic_block, stride, drop_ratio=0.):
        super().__init__()
        self.layer = self._make_layer(basic_block, in_channels, out_channels, stack_layers, stride, drop_ratio)

    def _make_layer(self, basic_block, in_channels, out_channels, stack_layers, stride, drop_ratio):
        layers = []
        for i in range(stack_layers):
            if i == 0:
                layers += [basic_block(in_channels, out_channels, stride, drop_ratio)]
            else:
                layers += [basic_block(out_channels, out_channels, 1, drop_ratio)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, h_args, widen_factor=1, drop_ratio=0.1, num_cls=8, use_init=True):
        super().__init__()
        in_c = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "depth should be 6n + 4"
        n = (depth - 4) // 6
        basic_block = BasicBlock
        self.conv1 = nn.Conv1d(3, in_c[0], kernel_size=3,
                               stride=1, padding=0)
        self.block1 = WrnBlock(n, in_c[0], in_c[1], basic_block, 1, drop_ratio)
        self.block2 = WrnBlock(n, in_c[1], in_c[2], basic_block, 2, drop_ratio)
        self.block3 = WrnBlock(n, in_c[2], in_c[3], basic_block, 2, drop_ratio)
        self.bn1 = nn.BatchNorm1d(in_c[3])
        self.act1 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.last_channels = in_c[3]
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

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        feat_m.append(self.classifier)
        return feat_m

    def forward(self, x, is_feat=False):
        b = x.shape[0]
        x = self.conv1(x)
        f0 = x
        x = self.block1(x)
        f1 = x
        x = self.block2(x)
        f2 = x
        x = self.block3(x)
        f3 = x
        x = self.avgpool(x)
        x = x.view(b, -1)
        f4 = x
        for module in self.classifier:
            x = module(x)
        if is_feat:
            return [f0, f1, f2, f3, f4], x
        else:
            return x

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


tensor = torch.rand(2, 3, 1000)


def wrn_40_2(h_args):
    model = WideResNet(depth=40, widen_factor=2, h_args=h_args)
    return model


def wrn_40_1(h_args):
    model = WideResNet(depth=40, widen_factor=1, h_args=h_args)
    return model


def wrn_16_2(h_args):
    model = WideResNet(depth=16, widen_factor=2, h_args=h_args)
    return model


def wrn_16_1(h_args):
    model = WideResNet(depth=16, widen_factor=1, h_args=h_args)
    return model
