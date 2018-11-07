import argparse


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='RecVis A3 training script')
        self.parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                                 help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
        self.parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                                 help='input batch size for training (default: 64)')
        self.parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                 help='number of epochs to train (default: 10)')
        self.parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                                 help='learning rate (default: 0.01)')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                 help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
        self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                 help='how many batches to wait before logging training status')
        self.parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                                 help='folder where experiment outputs are located.')

    def parse(self):
        return self.parser.parse_args()
