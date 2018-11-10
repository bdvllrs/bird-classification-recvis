from math import floor
import Augmentor
import torch
import torchvision.transforms as transforms
from models import bounding_box

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


def data_transformer(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
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
    def __init__(self, model, model_path):
        state_dict = torch.load(model_path)
        self.model, self.size = model
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __call__(self, img):
        vector_img = transforms.Resize(self.size)(img)
        vector_img = transforms.ToTensor()(vector_img)
        vector_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])(vector_img).unsqueeze(0)
        vector_img.requires_grad = False
        bbox = self.model(vector_img)[0].detach().numpy()
        return img.crop(get_rectangle(bbox, img.size))


def data_transformer_with_segmentation(size, model, model_path):
    return transforms.Compose([
        ResizeUsingBoudingBox(model, model_path),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def data_transformer_with_augment(input_size):
    p = Augmentor.Pipeline()
    p.rotate(probability=0.9, max_left_rotation=10, max_right_rotation=10)
    p.shear(probability=0.6, max_shear_left=10, max_shear_right=10)
    p.flip_random(probability=0.7)
    p.random_distortion(probability=0.5, grid_height=16, grid_width=16, magnitude=10)
    p.zoom_random(probability=0.7, percentage_area=0.8)
    return transforms.Compose([
        p.torch_transform(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
