from math import floor
import Augmentor
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F_vision
from torch.nn import functional as F
from typing import Tuple

from models import bounding_box


class Resize(transforms.Resize):
    def __call__(self, img):
        width, height = img.size
        if width > height:
            size = self.size if type(self.size) == int else self.size[0]
            size = (size, int(height / width * size))
        else:
            size = self.size if type(self.size) == int else self.size[1]
            size = (int(width / height * size), size)
        return F_vision.resize(img, size, self.interpolation)


class MinResize(transforms.Resize):
    """
    Resize so that all size are at least what is given and keeps proportions
    """

    def __call__(self, img):
        width, height = img.size
        size_width = self.size if type(self.size) == int else self.size[0]
        size_height = self.size if type(self.size) == int else self.size[1]
        if width < size_width:
            width, height = size_width, int(height * size_width / width)
        if height < size_height:
            width, height = int(width * size_height / height), size_width
        return F_vision.resize(img, (width, height), self.interpolation)


# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def data_transformer(size: Tuple[int, int], use_crop=False, no_resize=False, min_resize=None):
    """
    Give the transform function
    Args:
        size: size of resizing
        use_augmentation: if true, is augmentation
    Returns:

    """
    # def pad_for_square(img):
    #     width, height = img.size(1), img.size(2)
    #     if abs(width - height) % 2 == 0:
    #         pad = abs(width - height) // 2, abs(width - height) // 2
    #     else:
    #         pad = abs(width - height) // 2, abs(width - height) // 2 + 1
    #     if width > height:
    #         return F.pad(img, pad, 'constant', 0)
    #     else:
    #         pad = (0, 0,) + pad
    #         return F.pad(img, pad, 'constant', 0)

    # Use augmentor to randomly crop the image
    # p = Augmentor.Pipeline()
    # p.crop_by_size(1, 100, 100, centre=False)

    return transforms.Compose([
        # p.torch_transform() if use_crop else lambda x: x,
        # Resize(size),
        MinResize(min_resize) if min_resize is not None else lambda x: x,
        transforms.Resize(size) if not no_resize else lambda x: x,
        transforms.ToTensor(),
        # pad_for_square,
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_rectangle(bbox, input_size):
    left = min(bbox[0], bbox[1])
    right = max(bbox[0], bbox[1])
    bottom = min(bbox[2], bbox[3])
    top = max(bbox[3], bbox[3])
    left_x, bottom_y = input_size[0] * left, input_size[1] * bottom
    height, width = input_size[1] * (top - bottom), input_size[0] * (right - left)
    return left_x, bottom_y, left_x + width, bottom_y + height


class ResizeUsingBoudingBox(object):
    def __init__(self, model, model_path, final_size):
        state_dict = torch.load(model_path)
        self.model, self.size = model
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.final_size = final_size

    def __call__(self, img):
        vector_img = transforms.Resize(self.size)(img)
        vector_img = transforms.ToTensor()(vector_img)
        vector_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])(vector_img).unsqueeze(0)
        vector_img.requires_grad = False
        bbox = self.model(vector_img)[0].detach().numpy()
        cropped_img = img.crop(get_rectangle(bbox, img.size))
        cropped_img = transforms.Resize(self.size)(cropped_img)
        cropped_img = transforms.ToTensor()(cropped_img)
        cropped_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])(cropped_img)
        vector_img = vector_img.squeeze(dim=0)
        imgs = torch.empty(2, vector_img.size(0), vector_img.size(1), vector_img.size(2))
        imgs[0] = vector_img
        imgs[1] = cropped_img
        return imgs


def data_transformer_with_segmentation(size, model, model_path):
    return transforms.Compose([
        ResizeUsingBoudingBox(model, model_path, size),
    ])


def data_transformer_with_augment(input_size):
    p = Augmentor.Pipeline()
    p.random_erasing(probability=0.4, rectangle_area=0.2)
    p.rotate(probability=0.9, max_left_rotation=10, max_right_rotation=10)
    p.shear(probability=0.6, max_shear_left=10, max_shear_right=10)
    p.flip_random(probability=0.7)
    p.random_distortion(probability=0.5, grid_height=16, grid_width=16, magnitude=10)
    p.zoom(probability=0.7, min_factor=0.6, max_factor=1.5)
    return transforms.Compose([
        p.torch_transform(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class SegmentationImageLoader(ImageFolder):
    def __init__(self, root, **params):
        super(SegmentationImageLoader, self).__init__(root, **params)

    def __getitem__(self, index):
        path, target = self.samples[index]
        path_ori_image = path.replace('segmentations/', '').replace('.png', '.jpg')
        sample = self.loader(path_ori_image)
        sample_seg = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            sample_seg = self.transform(sample_seg)
        return sample, target, sample_seg
