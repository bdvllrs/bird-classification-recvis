import os
from datetime import datetime
import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from models import bounding_box, bbalexnet, unet11
from tools import Parser, data_transformer, SegmentationDataLoader, JacardLoss
from tools.visualisation import show_images, show_bounding_box, plot_error

model, input_size = bounding_box()
# model, input_size = unet11(pretrained=True)

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

path_to_images = os.path.abspath(os.path.join(os.curdir, 'bird_dataset/seg_dataset', 'train_images'))

# Data initialization and loading
data_transforms_train = data_transformer(input_size)
data_transforms_val = data_transformer(input_size)

train_loader = torch.utils.data.DataLoader(
    # Get the original image and set target as bounding box over segmentation
    SegmentationDataLoader(args.data + '/seg_dataset/segmentations/train_images',
                           transform=data_transforms_train),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    SegmentationDataLoader(args.data + '/seg_dataset/segmentations/val_images',
                           transform=data_transforms_val),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, fig_error, ax_error):
    model.train()
    error_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = JacardLoss()
        loss = criterion(output, target)
        loss.backward()
        error_loss += loss
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))

    plot_error(epoch, error_loss.detach().cpu().numpy()/len(train_loader), fig_error, ax_error)


def validation(epoch, fig_error, ax_error):
    model.eval()
    validation_loss = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # if epoch % 2 == 0:
        #     i = np.random.randint(0, len(output))
        #     fig, ax = plt.subplots(1)
        #     show_images(data, 3, min=i, max=i + 1, ax=ax)
        #     show_bounding_box(output[i], input_size, ax, color='r')
        #     show_bounding_box(target[i], input_size, ax, color='b')
        #     fig.show()
        # sum up batch loss
        # criterion = torch.nn.MSELoss(reduction='elementwise_mean')
        criterion = JacardLoss()
        validation_loss += criterion(output, target).data.item()
    validation_loss /= len(val_loader)
    plot_error(epoch, validation_loss, fig_error, ax_error)
    print('\nValidation set: Average loss: {})\n'.format(validation_loss))


path = args.experiment + '/' + datetime.today().strftime('%Y-%m-%d %H:%M:%S')
os.mkdir(path)
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Training error')
ax2.set_title('Validation error')
plt.ion()
fig.show()
for epoch in range(1, args.epochs + 1):
    train(epoch, fig, ax1)
    validation(epoch, fig, ax2)
    model_file = path + '/model.pth'
    torch.save(model.state_dict(), model_file)
    print(
        '\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file +
        '` to generate the Kaggle formatted csv file')
