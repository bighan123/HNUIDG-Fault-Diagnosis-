# The fault diagnosis code demo of HNU intelligent diagnosis Groups

#### 模型介绍

实现了经典CNN模型，视觉Transformer模型，Hybrid模型的智能故障诊断。处理的数据为一维振动数据，因此在相关模型的结构上（堆叠层数，参数，维度变换）与原作者论文有些许不同，具体实现的模型backbone如下：

| 经典CNN分类模型                     | 论文地址                                                     |
| :---------------------------------- | ------------------------------------------------------------ |
| VGG                                 | https://arxiv.org/abs/1409.1556                              |
| Mobilenetv2                         | https://arxiv.org/abs/1801.04381                             |
| Wrn                                 | https://arxiv.org/abs/1605.07146                             |
| ResNet                              | https://arxiv.org/abs/1512.03385                             |
| EHcnn (Proposed by HNU IDG)         | https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDAUTO&filename=HKXB202209008&uniplatform=NZKPT&v=8XwRD3UrBzc5RLf7bgtiV03xKtD_9kS4MV9A71YudCNH_8tQnvjpIXlFSqD3JoDc |
| Dilated EHcnn(Proposed by HNU IDG)  | https://iopscience.iop.org/article/10.1088/1361-6501/ac1b43  |
| **经典视觉Transformer模型**         |                                                              |
| ViT                                 | https://arxiv.org/abs/2010.11929                             |
| **Hybrid模型**                      |                                                              |
| Convformer-NSE(Proposed by HNU IDG) | https://ieeexplore.ieee.org/document/9872314                 |
| MaxVit                              | https://arxiv.org/abs/2204.01697                             |
| LocalVit                            | https://arxiv.org/abs/2104.05707                             |
| Neighborhood Attention Transformer  | https://arxiv.org/abs/2204.07143                             |

通过model_dict可以访问不同参数的backbone

```python
model_dict = {'vgg11': vgg11,
              'vgg13': vgg13,
              'vgg16': vgg16,
              'vgg19': vgg19,
              'convformer_v1_s': convoformer_v1_small,
              'convformer_v1_m': convoformer_v1_middle,
              'convformer_v1_b': convormer_v1_big,
              'convformer_v2_s': convoformer_v2_small,
              'convformer_v2_m': convoformer_v2_middle,
              'convformer_v2_b': convormer_v2_big,
              'wrn_16_1': wrn_16_1,
              'wrn_16_2': wrn_16_2,
              'wrn_40_1': wrn_40_1,
              'wrn_40_2': wrn_40_2,
              'ehcnn_24_16': ehcnn_24_16,
              'ehcnn_30_32': ehcnn_30_32,
              'ehcnn_24_16_dilation': ehcnn_24_16_dilation,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'resnet101': resnet101,
              'resnet152': resnet152,
              'vit_base': vit_base,
              'vit_middle_16': vit_middle_patch16,
              'vit_middle_32': vit_middle_patch32,
              "mobilenet_half": mobilenet_half,
              'max_vit_tiny_16': max_vit_tiny_16,
              'max_vit_tiny_32': max_vit_tiny_32,
              'max_vit_small_16': max_vit_small_16,
              'max_vit_small_32': max_vit_small_32,
              'localvit_base_patch16_type1': localvit_base_patch16_type1,
              'localvit_base_patch16_type2': localvit_base_patch16_type2,
              ' localvit_middle1_patch16_type1': localvit_middle1_patch16_type1,
              'localvit_middle12_patch16_type1': localvit_middle2_patch32_type1,
              'nat_tiny': nat_tiny,
              'nat_small':nat_small,
              'nat_base':nat_base}
```

#### 数据集介绍

1.湖南大学锥齿轮试验台故障数据

文件结构：

```
| Data/
|————work condition1.xx
|----work condition2.xx
|----.....
```

实验装置：

![湖南大学实验装置](https://github.com/bighan123/HNUIDG-Fault-Diagnosis-/blob/main/png/湖大试验台.jpg)



2.西安交通大学齿轮箱试验台故障数据

```python
| Data/
|----work condtion1
|    |---- Channel one.xx
|    |---- Channel two.xx
|    |---- ......
|----work condition2
|    |---- Channel one.xx
|    |---- Channel two.xx
|    |---- ......
|......
```

实验装置：

![西安交通大学实验装置](https://github.com/bighan123/HNUIDG-Fault-Diagnosis-/blob/main/png/Figure_XJTUGearbox.jpg)

3.DDS齿轮箱试验台故障数据

文件结构：

```python
| Data/
|---- work condtion1
|     |---- data.xx
|---- work condtion2
|     |---- data.xx
```

实验装置：

![DDS实验装置](https://github.com/bighan123/HNUIDG-Fault-Diagnosis-/blob/main/png/DDS.jpg)

#### 实验结果

实验采用了西安交通大学的齿轮箱公开数据集，每类故障训练样本为100个，测试样本为200个，样本长度为1024，双通道，连续两个样本之间的重合率为30%，实验结果如下：

| Model         | Type                          | Data length | Epochs | Best Top-1 Acc |
| ------------- | ----------------------------- | ----------- | ------ | -------------- |
| Vgg           | 'vgg11'                       | 1024        | 100    | 93.64%         |
| ResNet        | 'resnet18'                    | 1024        | 100    | 100%           |
| Ehcnn         | 'ehcnn_24_16'                 | 1024        | 100    | 100%           |
| Ehcnn_dilated | 'ehcnn_24_16_dilated'         | 1024        | 100    | 100%           |
| WRN           | 'wrn_16_1'                    | 1024        | 100    | 98.61%         |
| VIT           | 'vit_base'                    | 1024        | 100    | 77.72%         |
| Convformer    | 'convformer_v1_s'             | 1024        | 100    | 100%           |
| LocalVit      | 'localvit_base_patch16_type1' | 1024        | 100    | 100%           |
| MaxVit        | 'max_vit_tiny'                | 1024        | 100    | 88.13%         |
| Nat           | 'nat_tiny'                    | 1024        | 100    | 100%           |

#### 安装教程

代码是在Windows10，Python 3.7，Pytorch 1.7.01, CUDA10.1环境下进行测试

安装依赖库：

pip install -r requirement.txt

本地克隆代码：

git clone https://gitee.com/fletahsy/the-fault-diagnosis-code-demo-of-hnu-intelligent-diagnosis-team.git

#### 关键参数说明

```python
--optimizer_name: 支持使用的优化器,如果需要添加或自定义新的优化器,请修改create_optimizer函数
--lr_scheduler: 支持使用的学习率变化测率,如果需要添加或自定义新的策略,请修改create_scheduler函数
--loss_name: 支持使用的损失,如果需要添加或自定义新的损失函数,请修改creat_loss函数
--datasets: 支持使用的数据集,见数据集介绍三种文件结构
--model_name: 支持的Backbone, 见model_dict字典
--use_ratio: 是否采用ratio划分样本
--size: 每类别的总样本数,若use_ratio为True,则根据size和use_ratio划分训练样本和测试样本
--train_size_use:训练样本数,use_ratio为False时起作用,适用于不平衡数据集时的训练
--test_size:测试样本数,use_ratio为False时起作用,适用于不平衡数据集时的测试
--num_cls:分类类别
-ic, --input_channel:输入一维数据的channel数
--layer_args:分类层的结构参数
```

#### 如何使用

最简单的例子，指定work_dir, 模型和数据集

因为不同数据集对应的故障类别不同，也需要指定num_cls参数

```
python train.py --work_dir to/path/data --model vgg11 --datasets hnu_datasets --num_cls 8
```

当采用Vit，LocalVit，MaxVit训练时需要额外指定样本的长度（涉及到Patch Embed操作），样本的长度应该为32的整数倍

```
python train.py --work_dir to/path/data --model max_vit_tiny_16 --datasets hnu_datasets --length 1024 --num_cls 8
```

我们同样提供了train_dynamic.py文件用于训练（Proposed by HUN IDG）,适用于训练样本不平衡时对样本权重系数进行动态的调整。

Noted

代码目前只支持单GPU的训练和测试

#### 引用

如果你采用了EHcnn模型的代码作为对比实验，请引用：

```python
@article{Han2022DL,
        title={Intelligent fault diagnosis of aero-engine high-speed bearing using enhanced convolutional neural network},
        author={Han SongYu and Shao Haidong and Jiang Hongkai and Zhang Xiaoyang},
        journal={航空学报},
        year={2022}}
```

如果你采用了EHcnn模型或者enhanced cross entropy作为对比实验，请引用：

```
@article{Han2022DL,
        title={Novel multi-scale dilated CNN-LSTM for fault diagnosis of planetary gearbox with unbalanced samples under noisy environment},
        author={Han Songyu and Zhong Xiang and Shao Haidong and Xu Tianao and Zhao Rongding and Cheng Junsheng},
        journal={Measurement Science and Techonology},
        year={2021}}
```

如果你采用了Convformer-nse模型作为对比实验，请引用:

```
@article{Han2022DL,
        title={Convformer-NSE: A Novel End-to-End Gearbox Fault Diagnosis Framework Under Heavy Noise
Using Joint Global and Local Information},
        author={Han Songyu and Shao Haidong and Cheng Junsheng and Yang Xingkai and Cai Baoping},
        journal={IEEE/ASME Transactions on Mechatronics},
        year={2022}}
```

如果你采用了动态训练(train_dynamic.py)作为对比实验，请引用：

```
@article{Han2022DL,
        title={End-to-end chiller fault diagnosis using fused attention mechanism and dynamic cross-entropy under imbalanced datasets},
        author={Han SongYu and Shao Haidong and Huo Zhiqiang and Yang Xingkai and Cheng Junsheng},
        journal={Building and Environment},
        year={2022}}
```

如果你采用了公开的西安交通大学数据集（链接如下），请根据相关要求对论文进行引用，引用格式为：

```
[1] Tianfu Li, Zheng Zhou, Sinan Li, Chuang Sun, Ruqiang Yan, Xuefeng Chen, “The emerging graph 
neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study,”
*Mechanical Systems and Signal Processing*, vol. 168, pp. 108653, 2022. DOI:
10.1016/j.ymssp.2021.108653
```

[XJTU齿轮箱试验台数据集](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay)


#### 联系方式

如果对代码有任何问题，或者想要进行智能故障诊断，缺陷检测的交流，欢迎联系：

fletahsy@hnu.edu.cn

导师邮箱：hdshao@hnu.edu.cn

