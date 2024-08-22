import random
from torchvision.transforms import functional as F
from torchvision import transforms as transforms

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target

class ColorJitter(object):


    def __call__(self, image, target):
        image = transforms.ColorJitter(brightness=32/255, contrast=0.5, saturation=0.5, hue=0.5)(image)

        return image, target

class RandomErasing(object):


    def __call__(self, image, target):
        image = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)(image)

        return image, target

class RandomEqualize(object):


    def __call__(self, image, target):
        image = transforms.RandomEqualize(p=1)(image)

        return image, target


class GaussianBlur(object):


    def __call__(self, image, target):
        image = transforms.GaussianBlur(kernel_size=(5, 11),
                            sigma=(5, 10.0))(image)

        return image, target        

        
class RandomAdjustSharpness(object):


    def __call__(self, image, target):
        image = transforms.RandomAdjustSharpness(sharpness_factor=20,p=1)(image)

        return image, target        


class Normalize(object):


    def __call__(self, image, target):
        image = transforms.Normalize(mean=[0.5, ], std=[0.5, ])(image)

        return image, target            