import os
from datetime import datetime
import torch
import torch.optim as optim
from torchvision import datasets
from tools import Parser, data_transformer_with_segmentation
from tools.visualisation import show_images
from models import bounding_box
from models import simple_cnn, alexnet, resnet101_wo_softmax, LinearClassifier

import matplotlib.pyplot as plt

embedding_size = 100

model, input_size = resnet101_wo_softmax(embedding_size)
classifier = LinearClassifier(2*embedding_size)

# Training settings
args = Parser().parse()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

path_to_images = os.path.abspath(os.path.join(os.curdir, 'bird_dataset', 'train_images'))

# Data initialization and loading
data_transforms = data_transformer_with_segmentation(input_size, bounding_box(),
                                                     model_path='experiment/bb-v4/model.pth')

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in cnn.py so that it can be reused by the evaluate.py script

# model = SimpleCNN()
if use_cuda:
    print('Using GPU')
    model.cuda()
    classifier.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, data_cropped = data[:, 0], data[:, 1]
        if use_cuda:
            data, data_cropped, target = data.cuda(), data_cropped.cuda(), target.cuda()
        # show_images(data, 3, min=0, max=1)
        # plt.show()
        optimizer.zero_grad()
        emb1 = model(data)
        emb2 = model(data_cropped)
        emb = torch.cat((emb1, emb2), dim=-1)
        output = classifier(emb)
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
    for data, target in val_loader:
        data, data_cropped = data[:, 0], data[:, 1]
        if use_cuda:
            data, data_cropped, target = data.cuda(), data_cropped.cuda(), target.cuda()
        optimizer.zero_grad()
        emb1 = model(data)
        emb2 = model(data_cropped)
        emb = torch.cat((emb1, emb2), dim=-1)
        output = classifier(emb)
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
    print(
        '\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file +
        '` to generate the Kaggle formatted csv file')
