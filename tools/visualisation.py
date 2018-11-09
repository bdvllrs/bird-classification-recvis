import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision


def show_images(imgs, nrow=8, min=0, max=-1, ax=None):
    # Scale between [0, 1]
    imgs = imgs[min:max]
    imgs = (imgs - imgs.min().expand(imgs.size())) / (imgs.max() - imgs.min()).expand(imgs.size())
    # Convert to numpy image
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow)
    imgs = np.transpose(imgs.detach().cpu().numpy(), (1, 2, 0))
    plt.grid(False)
    plt.axis('off')
    if ax is not None:
        ax.grid(False)
        ax.axis('off')
        ax.imshow(imgs)
    else:
        plt.imshow(imgs)


def show_bounding_box(bbox, input_size, ax, color='r'):
    rect = patches.Rectangle((input_size[0] * bbox[0], input_size[1] * bbox[2]),
                             input_size[1] * (bbox[3] - bbox[2]), input_size[0] * (bbox[1] - bbox[0]),
                             linewidth=1, edgecolor=color, facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)


def plot_error(k, new_point, fig, ax):
    ax.plot(k, new_point)
    fig.canvas.draw()
    fig.canvas.flush_events()

