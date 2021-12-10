import torch

device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(0)

# data settings
dataset_path = "MVTec_AD"
class_name = "wood"
modelname = "wood_test"

img_size = (1024, 1024)
img_dims = [3] + list(img_size)

# transformation settings
transf_rotations = True     # 是否旋转
transf_brightness = 0.0     # 亮度度
transf_contrast = 0.0       # 对比度
transf_saturation = 0.0     # 饱和度
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # 标准化参数

# feature extractor
extractor_name = "resnet18"

# network hyperparameters
n_scales = 1 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp_alpha = 3 # see paper (differnet) equation 2 for explanation
n_coupling_blocks = 4
#fc_internal = 2048 # number of neurons in hidden layers of s-t-networks
dropout = 0.0 # dropout in s-t-networks
lr_init = 0.0002
subnet_conv_dim = 128

n_feat = 256 * n_scales

if(extractor_name == "resnet18"):
    n_feat = 64*64*64 * n_scales
if(extractor_name == "deit"):
    n_feat = 24*24*768 * n_scales

# dataloader parameters
transform_num = 4 # 训练中每个样本的转换数
transform_num_test = 64 # 测试中每个样本的转换数  

batch_size = 24 # actual batch size is this value multiplied by n_transforms(_test)
batch_size_test = batch_size * transform_num // transform_num_test

# Epoches数目
meta_epochs = 24
sub_epochs = 1   # 每隔几个epoch做一次评估

# 输出设置
verbose = True          # 详细资讯
grad_map_viz = False
hide_tqdm_bar = False
save_model = True       # 是否保存模型