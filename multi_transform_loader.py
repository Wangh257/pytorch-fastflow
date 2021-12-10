import torch
import torchvision.transforms as TF

from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import make_dataset, pil_loader, default_loader, IMG_EXTENSIONS
from torchvision.transforms.functional import rotate

import config as C

def fixed_rotation(self, sample, degrees):
    custom_rotate = lambda x: rotate(x, degrees, False, False, None) # 根据角度旋转图片
    custom_transforms = [TF.Resize(C.img_size)]

    augmentative_transforms = [custom_rotate]
    if C.transf_brightness > 0.0 or C.transf_contrast > 0.0 or C.transf_saturation > 0.0:
        color_transforms = TF.ColorJitter(
            brightness=C.transf_brightness,
            contrast=C.transf_contrast,
            saturation=C.transf_saturation
        ) # 调整图片的亮度、对比度、饱和度
        augmentative_transforms.append(color_transforms)

    custom_transforms.extend(augmentative_transforms)
    custom_transforms.append(TF.ToTensor())                         # 转化为Tensor
    custom_transforms.append(TF.Normalize(C.norm_mean, C.norm_std)) # 归一化

    return TF.Compose(custom_transforms)(sample)

class DatasetFolderMultiTransform(DatasetFolder):
    '''
    root: 根目录路径。
    loader: 由给定的路径加载数据。
    extensions: 允许的扩展。
    transform: 对输入数据进行处理，如: 对于图片有transforms.RandomCrop。
    target_transform: 对输入数据对应的target做处理，如旋转是要更新标注的目标位置信息。
    '''
    def __init__(self, 
        root: str, loader, extensions, 
        transform = None, target_transform = None, 
        is_valid_file = None, transform_num = 1
    ) -> None:
        super(DatasetFolderMultiTransform, self).__init__(root, 
            loader, extensions=extensions, 
            transform=transform, target_transform=target_transform, 
            is_valid_file=is_valid_file
        )

        classes, class_to_idx = self.find_classes(self.root)

        if is_valid_file is not None:
            extensions = None
        
        self.samples = make_dataset(self.root, class_to_idx, extensions)
        self.transform_num = transform_num
        self.get_fixed = False
        # 每个 transform 旋转不同的角度
        self.fixed_degrees = [i * 360.0 / transform_num for i in range(transform_num)] 
    
    def __getitem__(self, index):
        path, target = self.samples[index]      # 取出路径以及对应的类别 如 wood(木材)
        
        sample = self.loader(path)              # 通过路径加载样本 数组吗？

        if self.transform is not None:          # 做转换
            samples = []
            for i in range(self.transform_num):
                if self.get_fixed: 
                    samples.append(fixed_rotation(self, sample, self.fixed_degrees[i]))
                else: 
                    samples.append(self.transform(sample))

            samples = torch.stack(samples, dim=0) # 两个[Tensor[], Tensor[]] => Tensor[[], []]

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return samples, target

class ImageFolderMultiTransform(DatasetFolderMultiTransform):
    def __init__(self, 
        root: str, 
        loader=default_loader, 
        transform=None, target_transform=None, 
        is_valid_file=None, transform_num=C.transform_num
    ) -> None:
        super().__init__(root, 
            loader, IMG_EXTENSIONS, 
            transform=transform, target_transform=target_transform, 
            is_valid_file=is_valid_file, transform_num=transform_num
        )

        self.image_list = self.samples