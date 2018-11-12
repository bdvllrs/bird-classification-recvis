import os
from datetime import datetime
import torch
import torch.optim as optim
from torchvision import datasets
from tools import Parser, data_transformer, CNNLayerVisualization, SegmentationImageLoader
from models import simple_cnn, alexnet, resnet101, unet11
import matplotlib.pyplot as plt

model, input_size = resnet101()
seg_model = unet11(pretrained=True)

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

path_to_images = os.path.abspath(os.path.join(os.curdir, 'bird_dataset', 'train_images'))

# Data initialization and loading
data_transforms = data_transformer(input_size)

train_loader = torch.utils.data.DataLoader(
    SegmentationImageLoader(args.data + '/segmentations/train_images',
                            transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    SegmentationImageLoader(args.data + '/segmentations/val_images',
                            transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in cnn.py so that it can be reused by the evaluate.py script

# model = SimpleCNN()
if use_cuda:
    print('Using GPU')
    model.cuda()
    seg_model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target, seg_target) in enumerate(train_loader):
        if use_cuda:
            data, target, seg_target = data.cuda(), target.cuda(), seg_target.cuda()
        optimizer.zero_grad()
        segmentation = seg_model(data)
        seg = segmentation.expand_as(data)
        data.mul_(seg)
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target, seg_target in val_loader:
        if use_cuda:
            data, target, seg_target = data.cuda(), target.cuda(), seg_target.cuda()
        segmentation = seg_model(data)
        seg = segmentation.expand_as(data)
        data.mul_(seg)
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


path = args.experiment + '/' + datetime.today().strftime('%Y-%m-%d %H:%M:%S')
os.mkdir(path)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = path + '/model.pth'
    torch.save(model.state_dict(), model_file)
    seg_model_file = path + '/seg_model.pth'
    torch.save(seg_model.state_dict(), seg_model_file)
    print(
        '\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file +
        '` to generate the Kaggle formatted csv file')
