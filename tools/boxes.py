import numpy as np
import torch
from torch import Tensor as torch_Tensor
from torchvision.datasets import ImageFolder


def intersect(box_a, box_b):
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
    max_x1 = torch.max(box_a[:, 0], box_b[:, 0])
    min_x2 = torch.min(box_a[:, 1], box_b[:, 1])
    max_y1 = torch.max(box_a[:, 2], box_b[:, 2])
    min_y2 = torch.min(box_a[:, 3], box_b[:, 3])
    inter = torch.clamp((min_x2 - max_x1), min=0) * torch.clamp((min_y2 - max_y1), min=0)
    return inter


def jaccard(box_a, box_b, smooth=1):
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
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 1] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 2]))
    area_b = ((box_b[:, 1] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 2]))
    union = area_a + area_b - inter + smooth
    return (inter + smooth) / union  # [A,B]


class JacardLoss(torch.nn.Module):
    def forward(self, bbox1, bbox2):
        smooth = 10
        return torch.sum((1 - jaccard(bbox1, bbox2, smooth)) * smooth)


def bounding_box(img):
    """

    Args:
        img:

    Returns: height_min, height_max, width_min, width_max

    """
    if type(img) == torch_Tensor:
        img = img.numpy()
    img = img[0, :, :] - np.expand_dims(img[0, 0, 0], 1)  # remove the top left pixel and set it as background
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin / len(img), rmax / len(img), cmin / len(img[0]), cmax / len(img[0])


class SegmentationDataLoader(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        path_ori_image = path.replace('segmentations/', '').replace('.png', '.jpg')
        sample = self.loader(path_ori_image)
        sample_seg = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            sample_seg = self.transform(sample_seg)
        target = torch_Tensor(bounding_box(sample_seg))
        return sample, target
