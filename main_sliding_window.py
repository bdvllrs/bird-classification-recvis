import os
from datetime import datetime
import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from models import bounding_box, bbalexnet, resnet101
from tools import Parser, data_transformer, SegmentationDataLoader, JacardLoss, dice_coeff, grids, \
    bounding_box as bounding_box_f
from tools.visualisation import show_images, show_bounding_box, plot_error

# model, input_size = bounding_box()
model, input_size = resnet101(nclass=21)

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

path_to_images = os.path.abspath(os.path.join(os.curdir, 'bird_dataset/seg_dataset', 'train_images'))

# Data initialization and loading
data_transforms_train = data_transformer(input_size, use_crop=True, min_resize=input_size)
data_transforms_val = data_transformer(input_size, use_crop=True, no_resize=True, min_resize=input_size)

sliding_window = input_size[0]

train_loader = torch.utils.data.DataLoader(
    # Get the original image and set target as bounding box over segmentation
    SegmentationDataLoader(args.data + '/segmentations/train_images',
                           transform=data_transforms_train, sliding_windows=sliding_window),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    SegmentationDataLoader(args.data + '/segmentations/val_images',
                           transform=data_transforms_val, sliding_windows=sliding_window, validation=True),
    batch_size=1, shuffle=False, num_workers=1)

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, fig_error, ax_error):
    model.train()
    for batch_idx, (ori, data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))


def validation(epoch, fig_error, ax_error):
    model.eval()
    validation_loss = 0
    correct = 0
    for seg, data, target in val_loader:
        boxes = grids(sliding_window, data.size(2), data.size(3))
        # Copy
        new_data = torch.zeros(len(boxes), data.size(1), sliding_window, sliding_window)
        new_targets = torch.zeros(len(boxes), dtype=torch.long)
        for k, box in enumerate(boxes):
            new_data[k, :, :, :] = data[0, :, box[0]:box[1], box[2]:box[3]]
            bbox = bounding_box_f(seg[0, :, box[0]:box[1], box[2]:box[3]])
            new_targets[k] = target[0] if bbox is not None else 20  # Background if not bird

        if use_cuda:
            new_data, new_targets = new_data.cuda(), new_targets.cuda()
        output = model(new_data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, new_targets).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        # Choose by averaging
        found_classes = {}
        for k, target_class in enumerate(pred):
            val_class = int(target_class[0].detach().cpu().numpy())
            if val_class != 20:  # If not background
                if val_class not in found_classes.keys():
                    found_classes[val_class] = float(output[k][val_class].detach().cpu().numpy())
                else:
                    found_classes[val_class] = max(found_classes[val_class],
                                                   float(output[k][val_class].detach().cpu().numpy()))
        if len(found_classes.keys()) == 0:
            pred = 0  # If nothing, we randmoly select
        else:
            pred = max(found_classes.items(), key=lambda x: x[1])[0]
        correct += pred == int(target.detach().cpu().numpy())
        # validation_loss += dice_coeff(output, new_targets).data.item()
    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


path = args.experiment + '/' + datetime.today().strftime('%Y-%m-%d %H:%M:%S')
os.mkdir(path)
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Training error')
ax2.set_title('Validation error')
# plt.ion()
# fig.show()
for epoch in range(1, args.epochs + 1):
    train(epoch, fig, ax1)
    validation(epoch, fig, ax2)
    model_file = path + '/model.pth'
    torch.save(model.state_dict(), model_file)
    print(
        '\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file +
        '` to generate the Kaggle formatted csv file')
