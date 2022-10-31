from .vgg import vgg11, vgg13, vgg16, vgg19
from .mobilev2 import mobilenet_half
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .ehcnn import ehcnn_24_16, ehcnn_30_32, ehcnn_24_16_dilation
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vit import vit_base, vit_middle_patch16, vit_middle_patch32
from .maxvit import max_vit_tiny_16, max_vit_tiny_32, max_vit_small_16, max_vit_small_32
from .localvit import localvit_base_patch16_type1, localvit_base_patch16_type2, localvit_middle1_patch16_type1, \
    localvit_middle2_patch32_type1
from .convformerv1 import convoformer_v1_small, convoformer_v1_middle, convormer_v1_big
from .convformerv2 import convoformer_v2_small, convoformer_v2_middle, convormer_v2_big
from .nat import nat_tiny, nat_small, nat_base

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
