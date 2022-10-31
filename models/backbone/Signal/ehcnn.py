import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The pytorch version of enhanced convolutional neural network 
Original Author: SongYu Han
Reported by "Intelligent fault diagnosis of aero-engine high-speed bearing uing enhanced convolutional neural network
doi:10.7527/S1000-6893.2021.25479

'''


class Baisc_cnn(nn.Module):
    def __init__(self, stack_layers, kernel_size, pool_size, in_channels, out_channels, dr=1):
        super().__init__()
        self.stack_layers = stack_layers
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.in_c = in_channels
        self.dr = dr
        self.out_c = out_channels
        self.layer = self._make_layer()
    def _make_layer(self):
        layers = []
        for i in range(self.stack_layers):
            if i == 0:
                layers += [
                    nn.Conv1d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=self.kernel_size,
                              stride=1, padding=int(self.kernel_size / 2)),
                    nn.Hardswish()]
            else:
                layers += [
                    nn.Conv1d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=self.kernel_size,
                              stride=1, padding=int(self.kernel_size / 2)),
                    nn.Hardswish()]
            if self.dr > 1:
                if i % 2 == 0:
                    idx = int(i * 2)
                    layers[idx].dilation = (self.dr,)
                    layers[idx].padding = (layers[idx].kernel_size[0] + (layers[idx].kernel_size[0] - 1) *  (self.dr - 1)) // 2

        layers += [nn.AvgPool1d(kernel_size=self.pool_size, padding=0)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class weight_Fuse(nn.Module):
    def __init__(self, in_channels, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=in_channels // 2,
                               kernel_size=kernel_size,
                               padding=int(kernel_size / 2))
        self.Max = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=in_channels // 2,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               padding=int(kernel_size / 2))

    def forward(self, x):
        b, c, n = x.shape
        v = x
        # [b, c, n] ==> [b, c//2, n]
        weight = self.conv1(x)
        # [b, c//2, n] ==> [b, c//2, n//2]
        weight = self.Max(weight)
        # [b, c//2, n//2] ==> [b, c, n//2]
        weight = self.conv2(weight)
        weight = F.interpolate(weight, size=n)
        return weight * x


class Ehcnn(nn.Module):
    def __init__(self, cfgs, h_args, in_c, dr=1, num_cls=8):
        super().__init__()
        self.in_c = in_c
        self.dr = dr
        self.parallel1, self.short_cut1, out_c1 = self._make_layer(cfgs[0])
        self.parallel2, self.short_cut2, out_c2 = self._make_layer(cfgs[1])
        self.parallel3, self.short_cut3, out_c3 = self._make_layer(cfgs[2])
        self.last_channels = out_c1 + out_c2 + out_c3
        self.weight_fuse = weight_Fuse(in_channels=self.last_channels)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
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

    def forward(self, x):
        b = x.shape[0]
        f1 = self.parallel1(x) + self.short_cut1(x)
        f2 = self.parallel2(x) + self.short_cut2(x)
        f3 = self.parallel3(x) + self.short_cut3(x)
        fused_feature = torch.cat([f1, f2, f3], dim=1)
        out = self.weight_fuse(fused_feature)
        out = self.avg_pool(out)
        out = out.view(b, -1)
        for module in self.classifier:
            out = module(out)
        return out

    def _make_layer(self, cfgs):
        cnn_layers = []
        short_cut_layers = []
        in_c = self.in_c
        short_cut_pool = 1
        for params in cfgs:
            cnn_layers += [Baisc_cnn(stack_layers=params[0],
                                     kernel_size=params[1],
                                     pool_size=params[2],
                                     in_channels=in_c,
                                     out_channels=params[3],
                                     dr=self.dr)]
            short_cut_pool *= params[2]
            in_c = params[3]
        short_cut_layers += [Baisc_cnn(stack_layers=2,
                                       kernel_size=1,
                                       pool_size=short_cut_pool,
                                       in_channels=self.in_c,
                                       out_channels=in_c,
                                       )]
        return nn.Sequential(*cnn_layers), nn.Sequential(*short_cut_layers), in_c


cfgs = {"ehcnn_24_16": [[[2, 13, 4, 8], [2, 13, 2, 16], [2, 13, 2, 32]],
                        [[2, 7, 4, 8], [2, 7, 2, 16], [2, 7, 2, 32]],
                        [[2, 3, 4, 8], [2, 3, 2, 16], [2, 3, 2, 32]]],
        "ehcnn_30_32": [[[3, 17, 4, 16], [3, 17, 4, 32], [3, 17, 2, 64]],
                        [[3, 11, 4, 16], [3, 11, 4, 32], [3, 11, 4, 64]],
                        [[3, 3, 4, 16], [3, 3, 4, 32], [3, 3, 4, 64]]]}


def ehcnn_24_16(h_args, in_c, num_cls):
    model = Ehcnn(cfgs=cfgs["ehcnn_24_16"],
                  h_args=h_args,
                  in_c=in_c,
                  num_cls=num_cls)
    return model


def ehcnn_30_32(h_args, in_c, num_cls):
    model = Ehcnn(cfgs=cfgs["ehcnn_30_32"],
                  h_args=h_args,
                  in_c=in_c,
                  num_cls=num_cls)
    return model

def ehcnn_24_16_dilation(h_args,
                         in_c,
                         num_cls,
                         dr = 3):
    model = Ehcnn(cfgs=cfgs["ehcnn_24_16"], h_args=h_args, in_c=in_c,num_cls=num_cls, dr=dr)
    return model

if __name__ == "__main__":
    tensor = torch.rand(2, 2, 1024)
    model = ehcnn_24_16_dilation(h_args=None, in_c=2, num_cls=9)
    print(model(tensor).shape)
