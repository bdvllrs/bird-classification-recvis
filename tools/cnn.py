from math import floor
from typing import Tuple, List, Union
from torch.nn import Conv2d, MaxPool2d

conv_or_pool_t = Union[Conv2d, MaxPool2d]


def output_size_conv2_layer(height: int, width: int, layer: conv_or_pool_t) -> Tuple[int, int]:
    """
    Compute output size from a conv2d layer
    Args:
        height: height of the image
        width: width of the image
        layer: convolution layer

    Returns: (height_out, width_out)

    """
    kernel_size, stride, padding, dilation = layer.kernel_size, layer.stride, layer.padding, layer.dilation
    height_out = floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    width_out = floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return height_out, width_out


def output_size_seq_conv2_layer(height: int, width: int, sequence: List[conv_or_pool_t]) -> Tuple[int, int]:
    """
    Get output height and width from a sequence of conv2d
    Args:
        height:
        width:
        sequence:

    Returns:

    """
    size_seq = len(sequence)
    height_out, width_out = height, width
    for k in range(size_seq):
        height_out, width_out = output_size_conv2_layer(height_out, width_out, sequence[k])
    return height_out, width_out
