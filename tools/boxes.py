import numpy as np
import torch
from torch import Tensor as torch_Tensor
from torchvision.datasets import ImageFolder
from torch.autograd import Function, Variable


def intersect_bbox(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    # print(box_a.size())
    # print(box_b.size())
    left_x_a = torch.min(box_a[:, 0], box_a[:, 1])
    right_x_a = torch.max(box_a[:, 0], box_a[:, 1])
    left_y_a = torch.min(box_a[:, 2], box_a[:, 3])
    right_y_a = torch.max(box_a[:, 3], box_a[:, 3])
    left_x_b = torch.min(box_b[:, 0], box_b[:, 1])
    right_x_b = torch.max(box_b[:, 0], box_b[:, 1])
    left_y_b = torch.min(box_b[:, 2], box_b[:, 3])
    right_y_b = torch.max(box_b[:, 3], box_b[:, 3])
    max_x1 = torch.max(left_x_a, left_x_b)
    min_x2 = torch.min(right_x_a, right_x_b)
    max_y1 = torch.max(left_y_a, left_y_b)
    min_y2 = torch.min(right_y_a, right_y_b)
    inter = torch.clamp((min_x2 - max_x1), min=0) * torch.clamp((min_y2 - max_y1), min=0)
    return inter


def jaccard_bbox(box_a, box_b, smooth=1):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        smooth:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect_bbox(box_a, box_b)
    area_a = (torch.abs(box_a[:, 1] - box_a[:, 0]) *
              torch.abs(box_a[:, 3] - box_a[:, 2]))
    area_b = (torch.abs(box_b[:, 1] - box_b[:, 0]) *
              torch.abs(box_b[:, 3] - box_b[:, 2]))
    union = area_a + area_b - inter + smooth
    return (inter + smooth) / union


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class JacardLoss(torch.nn.Module):
    def forward(self, bbox1, bbox2):
        smooth = 1
        return (1 - torch.mean(jaccard_bbox(bbox1, bbox2, smooth))) * smooth


def bounding_box(img):
    """

    Args:
        img:

    Returns: height_min, height_max, width_min, width_max

    """
    if type(img) == torch_Tensor:
        img = img.numpy()
    try:
        img = img[0, :, :] - np.expand_dims(img[0, 0, 0], 1)  # remove the top left pixel and set it as background
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin / len(img), rmax / len(img), cmin / len(img[0]), cmax / len(img[0])
    except:
        return None


def bbox_area(bbox):
    return (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])


def grids(size, width, height):
    """
    Returns all possible boxes of the image
    Args:
        size: size of the boxes
        width: width of the image
        height: height of the image
    """
    boxes = []
    for x in range(0, width-size, size):
        for y in range(0, height-size, size):
            boxes.append((x, x+size, y, y+size))
    for y in range(0, height-size, size):
        boxes.append((width-size, width, y, y+size))
    for x in range(0, width-size, size):
        boxes.append((x, x+size, height-size, height))
    return boxes


def random_box(size, width, height):
    x, y = int(np.random.rand() * (width - size)), int(np.random.rand() * (height - size))
    return x, x + size, y, y + size


class SegmentationDataLoader(ImageFolder):
    def __init__(self, root, bbox=True, sliding_windows=False, validation=False, **params):
        super(SegmentationDataLoader, self).__init__(root, **params)
        self.sliding_windows = sliding_windows
        self.bbox = bbox
        self.validation = validation

    def __getitem__(self, index):
        path, target = self.samples[index]
        path_ori_image = path.replace('segmentations/', '').replace('.png', '.jpg')
        sample = self.loader(path_ori_image)
        sample_seg = self.loader(path)
        if self.sliding_windows and not self.validation:
            crop_box = random_box(self.sliding_windows, sample.size[0], sample.size[1])
            sample_seg = sample_seg.crop((crop_box[0], crop_box[2], crop_box[1], crop_box[3]))
            sample = sample.crop((crop_box[0], crop_box[2], crop_box[1], crop_box[3]))

        if self.transform is not None:
            sample = self.transform(sample)
            sample_seg = self.transform(sample_seg)
        if self.sliding_windows:
            if not self.validation:
                bbox = bounding_box(sample_seg)
                target = target if bbox is not None else 20  # Background if not bird
                if bbox is not None and bbox_area(bbox) < 0.1:  # If less than 10% of area is bird, it's background
                    target = 20
            return sample_seg, sample, target
        elif self.bbox:
            target = torch_Tensor(bounding_box(sample_seg))
            return sample, target
        return sample, sample_seg
