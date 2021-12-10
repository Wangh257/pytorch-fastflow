import os
import torch

import torchvision.transforms.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from multi_transform_loader import ImageFolderMultiTransform

from torch import Tensor

import config as C

def get_random_transforms():
    train_transforms = [transforms.Resize(C.img_size)] # 图片大小重置
    augmentative_transforms = []
    if C.transf_rotations:
        augmentative_transforms.append(transforms.RandomRotation(180))  # 旋转180度
    if C.transf_brightness > 0.0 or C.transf_contrast > 0.0 or C.transf_saturation > 0.0:
        color_transforms = transforms.ColorJitter(
            brightness=C.transf_brightness, 
            contrast=C.transf_contrast,
            saturation=C.transf_saturation
        ) # 调整亮度、对比度、饱和度
        augmentative_transforms.append(color_transforms)

    train_transforms.extend(augmentative_transforms)
    train_transforms.extend([
        transforms.ToTensor(),  # 转化为Tensor
        transforms.Normalize(C.norm_mean, C.norm_std)   # 归一化
    ])

    return transforms.Compose(train_transforms)

def tensor2numpy(tensor: Tensor):
    return tensor.cpu().data.numpy if tensor is not None else None

'''
这个LOSS是如何定义出来的？？？？？？？
'''
def calcute_loss(z: Tensor, jacobian):
    # 计算每个样本的像素值开方之后的一半 z.shape = n * C*H*W
    result = 0.5 * torch.sum(z ** 2, dim=(1,2,3)) # result.shape = n
    # 值减去 jacobian jacobian.shape = n 后取均值 除以样本通道？
    result = torch.mean(result - jacobian) / z.size(1)

    return result

def load_datasets(dataset_path, class_name):
    class_perm = []

    target_transform = lambda target: class_perm[target]
    
    data_dir_train = os.path.join(dataset_path, class_name, 'train') # 训练数据地址
    data_dir_test = os.path.join(dataset_path, class_name, 'test')   # 测试数据地址
    classes = os.listdir(data_dir_test)                              # 取出所有的类别

    '''不做数据集的校验，请保证数据完整正确，测试数据集也需要包含 good '''

    classes.sort()      # 意义何在？
    class_idx = 1

    for zlass in classes:
        if zlass == 'good': class_perm.append(0) # 使用0表示正常样本
        else: class_perm.append(class_idx)
        class_idx += 1
    
    train_transform = get_random_transforms()

    train_dataset = ImageFolderMultiTransform(data_dir_train, transform=train_transform, transform_num=C.transform_num)
    test_dataset = ImageFolderMultiTransform(data_dir_test, 
        transform=train_transform, 
        target_transform=target_transform,
        transform_num=C.transform_num
    )

    return train_dataset, test_dataset

def make_dataloaders(train_dataset, test_dataset):
    trainloader = DataLoader(
        train_dataset, 
        pin_memory=True,            # 当计算机的内存充足的时候，可以设置pin_memory=True
        batch_size=C.batch_size, 
        shuffle=True                # 重新洗牌
    )
    testloader = DataLoader(
        test_dataset, 
        pin_memory=True, 
        batch_size=C.batch_size_test, 
        shuffle=True
    )
    return trainloader, testloader


def preprocess_batch(data):
    '''move data to device and reshape image'''
    inputs, labels = data
    inputs, labels = inputs.to(C.device), labels.to(C.device)
    inputs = inputs.view(-1, *inputs.shape[-3:])                # 干了什么？

    return inputs, labels