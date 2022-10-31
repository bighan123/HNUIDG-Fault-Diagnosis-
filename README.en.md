# The fault diagnosis repository of HNU intelligent diagnosis Groups

#### Description of model

The classical CNN,  Vision Transformer and Hybrid model is supported for intelligent fault diagnosis. Our repository is based on processing one dimensional data, so we made some change on the structure and hyper parameters of the model compared with the original settings. The detailed information about backbone are as follows

| classical CNN model                 | URL                                                          |
| :---------------------------------- | ------------------------------------------------------------ |
| VGG                                 | https://arxiv.org/abs/1409.1556                              |
| Mobilenetv2                         | https://arxiv.org/abs/1801.04381                             |
| Wrn                                 | https://arxiv.org/abs/1605.07146                             |
| ResNet                              | https://arxiv.org/abs/1512.03385                             |
| EHcnn (Proposed by HNU IDG)         | https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDAUTO&filename=HKXB202209008&uniplatform=NZKPT&v=8XwRD3UrBzc5RLf7bgtiV03xKtD_9kS4MV9A71YudCNH_8tQnvjpIXlFSqD3JoDc |
| Dilated EHcnn(Proposed by HNU IDG)  | https://iopscience.iop.org/article/10.1088/1361-6501/ac1b43  |
| **Vision Transformer model**        |                                                              |
| ViT                                 | https://arxiv.org/abs/2010.11929                             |
| **Hybrid model**                    |                                                              |
| Convformer-NSE(Proposed by HNU IDG) | https://ieeexplore.ieee.org/document/9872314                 |
| MaxVit                              | https://arxiv.org/abs/2204.01697                             |
| LocalVit                            | https://arxiv.org/abs/2104.05707                             |
| Neighborhood Attention Transformer  | https://arxiv.org/abs/2204.07143                             |

The model_dict is given to obtain model with different hyper parameters

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

#### Description of datasets

1.The bevel gearbox fault datasets collect from HNU

Data file structure：

```
| Data/
|————work condition1.xx
|----work condition2.xx
|----.....
```

Experimental rig ：

![b22907551611ab9322ce02788af09ce](D:\深度学习\测试数据\湖大螺旋锥齿轮新箱体test\试验台\b22907551611ab9322ce02788af09ce.jpg)

2.The gearbox fault datasets collect from XJTU

Data file structure：

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

Experimental rig ：

![image-20221028142056787](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221028142056787.png)

3.The gearbox fault datasets collect from DDS

Data file structure：

```python
| Data/
|---- work condtion1
|     |---- data.xx
|---- work condtion2
|     |---- data.xx
```

Experimental rig ：

![image-20221028142239324](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221028142239324.png)

#### Experimental results

The gearbox fault datasets collect from XJTU (open access) is used to conduct experiment, with 100 training samples and 200 test samples for each type of work condition.  Each sample is composed of 1024 points and two channels, and the overlap rate is 30% between two consecutive samples. the experimental results are shown in following table. 

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

#### Installation Tutorial 

This repository is tested on Windows 10, Python 3.7, Pytorch 1.7.01 and CUDA 10.1

Installing dependency repository:

pip install -r requirement.txt

Download code locally with git：

git clone https://gitee.com/fletahsy/the-fault-diagnosis-code-demo-of-hnu-intelligent-diagnosis-team.git

#### Description of key parameters

```
--optimizer_name: str, The optimizer which is supported in training processing, if you want to add or define new optimzier, please update the create_optimizer function
--lr_scheduler: str, The learning rate scheduler which is supported in training processing, if you want to add or define new scheduler, please update the create_scheduler function
--loss_name: str, The loss function which is supported in training processing, if you want to add or define new loss, please update the create_loss function
--datasets: str, The way which is supported to load datasets
--model_name: str, the model in repository
--use_ratio: bool, whether to use ratio to divide datasets in traning and testing samples
--size: The total number of samples in each category. It works only the use_ratio is set as True
--train_size_use: The number of training samples of each category. It works only the use_ratio is set as False
--test_size:The number of training samples of each category. It works only the use_ratio is set as False
--num_cls: The number of categories
-ic, --input_channel: The channel of the input data
--layer_args: The structure and parameters of classifier
```

#### How to use

For the simplest example, specify work_dir, model and dataset

Because different datasets correspond to different fault classes, you also need to specify the num_cls parameter

```
python train.py --work_dir to/path/data --model vgg11 --datasets hnu_datasets --num_cls 8
```

For example, when using Vit, LocalVit training, there is need to additionally specify the length of the sample (involving Patch Embed), the length of the sample should be an integer multiple of 32

```
python train.py --work_dir to/path/data --model max_vit_tiny_16 --datasets hnu_datasets --length 1024 --num_cls 8
```

We also provide the train_dynamic.py file for training (Proposed by HUN IDG), which is suitable for dynamic adjustment of the sample weight coefficients when the training samples are unbalanced.

Noted

The code currently only supports single GPU training and testing

#### Citation

If you used the the EHcnn model as a comparison model, please cite:

```python
@article{Han2022DL,
        title={Intelligent fault diagnosis of aero-engine high-speed bearing using enhanced convolutional neural network},
        author={Han SongYu and Shao Haidong and Jiang Hongkai and Zhang Xiaoyang},
        journal={航空学报},
        year={2022}}
```

If you used the EHcnn_dilation model or enhanced cross entropy as a comparison model, please cite:

```
@article{Han2022DL,
        title={Novel multi-scale dilated CNN-LSTM for fault diagnosis of planetary gearbox with unbalanced samples under noisy environment},
        author={Han Songyu and Zhong Xiang and Shao Haidong and Xu Tianao and Zhao Rongding and Cheng Junsheng},
        journal={Measurement Science and Techonology},
        year={2021}}
```

If you have used the Convformer-nse model as a comparison model, please cite:

```
@article{Han2022DL,
        title={Convformer-NSE: A Novel End-to-End Gearbox Fault Diagnosis Framework Under Heavy Noise
Using Joint Global and Local Information},
        author={Han Songyu and Shao Haidong and Cheng Junsheng and Yang Xingkai and Cai Baoping},
        journal={IEEE/ASME Transactions on Mechatronics},
        year={2022}}
```

If you used dynamic training (train_dynamic.py) as a comparison experiment, please cite:

```
@article{Han2022DL,
        title={End-to-end chiller fault diagnosis using fused attention mechanism and dynamic cross-entropy under imbalanced datasets},
        author={Han SongYu and Shao Haidong and Huo Zhiqiang and Yang Xingkai and Cheng Junsheng},
        journal={Building and Environment},
        year={2022}}
```

If you have used the publicly  dataset from XJTU , please cite:

```
[1] Tianfu Li, Zheng Zhou, Sinan Li, Chuang Sun, Ruqiang Yan, Xuefeng Chen, “The emerging graph 
neural networks for intelligent fault diagnostics and prognostics: A guideline and a benchmark study,”
*Mechanical Systems and Signal Processing*, vol. 168, pp. 108653, 2022. DOI:
10.1016/j.ymssp.2021.108653
```

[XJTU Gearbox Datasets](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay)


#### Contact

If you have any questions about the codes or would like to communicate about intelligent fault diagnosis, fault detection,please contact us.

fletahsy@hnu.edu.cn

Mentor E-mail：hdshao@hnu.edu.cn

