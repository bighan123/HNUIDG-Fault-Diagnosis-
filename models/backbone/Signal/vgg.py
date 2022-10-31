import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'训练图片数据时可以导入Imagnet的预训练权重'
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class Vgg_1d(nn.Module):
    def __init__(self, cfg, in_c, h_args=None, use_bn=False, use_init=True, num_cls=8):
        super().__init__()
        self.stage1 = self._make_layers(cfg[0], use_bn, in_c)
        self.stage2 = self._make_layers(cfg[1], use_bn, cfg[0][-2])
        self.stage3 = self._make_layers(cfg[2], use_bn, cfg[1][-2])
        self.stage4 = self._make_layers(cfg[3], use_bn, cfg[2][-2])
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.ModuleList()
        if not h_args:
            self.classifier.append(nn.Linear(512, num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        else:
            for i in range(len(h_args)):
                if i == 0:
                    self.classifier.append(nn.Linear(512, h_args[i]))
                else:
                    self.classifier.append(nn.Linear(h_args[i - 1], h_args[i]))
            self.classifier.append(nn.Linear(h_args[-1], num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        if use_init:
            self._initialize_weights()

    def forward(self, x, is_feat=False):
        x = self.stage1(x)
        f0 = x
        x = self.stage2(x)
        f1 = x
        x = self.stage3(x)
        f2 = x
        x = self.stage4(x)
        f3 = x
        x = self.avg(x)
        f4 = x
        x = x.view(-1, 512)
        for block in self.classifier:
            x = block(x)
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
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _make_layers(cfg, use_bn=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                cnn_layers = nn.Conv1d(in_channels=in_channels,
                                       out_channels=v,
                                       kernel_size=3,
                                       padding=1)
                if use_bn:
                    layers += [cnn_layers, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [cnn_layers, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)


cfgs = {
    'vgg11': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'vgg13': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'vgg16': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
    'vgg19': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'],
              [512, 512, 512, 512, 'M']],
}


def vgg11(h_args, in_c, num_cls):
    model = Vgg_1d(cfg=cfgs["vgg11"], h_args=h_args,
                   in_c=in_c, num_cls=num_cls)
    return model


def vgg13(h_args, in_c, num_cls):
    model = Vgg_1d(cfg=cfgs['vgg13'], h_args=h_args,
                   in_c=in_c, num_cls=num_cls)
    return model


def vgg16(h_args, in_c, num_cls):
    model = Vgg_1d(cfg=cfgs['vgg16'], h_args=h_args,
                   in_c=in_c, num_cls=num_cls)
    return model


def vgg19(h_args, in_c, num_cls):
    model = Vgg_1d(cfg=cfgs['vgg19'], h_args=h_args,
                   in_c=in_c, num_cls=num_cls)
    return model

if __name__ == "__main__":
    tensor = torch.rand(4, 8, 1024)
    model = vgg19(h_args=None, in_c=8)
    print(model(tensor).shape)