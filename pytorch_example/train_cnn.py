
import time
from loader import GCommandLoader


import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F

import sys
import os
import glob

import numpy as np

batch_size = 1
num_epochs = 1


class CNN(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self,drop_out):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5,stride=1)
        self.conv2_drop = nn.Dropout2d(p = drop_out)
        self.fc1 = nn.Linear(3080, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



def train(train_path,cnn,loss_func,optimizer):
    print('Loading data...')
    start_time = time.time()
    train_dataset = GCommandLoader(train_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
    num_workers=20)

    global_epoch_loss = 0
    cnn.train()
    for step, (data, target) in enumerate(train_loader):
        # print data.shape
        # print target.shape
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = cnn(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        global_epoch_loss += loss.data[0]
        if step % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            num_epochs, step * len(data), len(train_loader.dataset), 100. * step / len(train_loader), loss.data[0]))


    return global_epoch_loss / len(train_loader.dataset)

def test(test_path,cnn,loss_func):
    test_dataset = GCommandLoader(test_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
    num_workers=20)

    cnn.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader :

        data, target = Variable(data, requires_grad=False), Variable(target)
        output = cnn(data)
        test_loss += loss_func(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / float(len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

    return test_loss,accuracy




if __name__ == "__main__":


    train_path = "redimension/train"

    test_path = "redimension/test"

    epoch = 1
    patience = 3
    best_loss = np.inf
    iteration = 0
    learning_rate = [0.01,0.1,0.5]
    opt = ['Adam','SGD']
    drop_out = [0.05,0.1,0.2]

    while (epoch < num_epochs + 1) and (iteration < patience):
        for do in drop_out:
            cnn = CNN(do)
            loss_func = nn.CrossEntropyLoss()
            for lr in learning_rate:
                for op in opt:
                    if op == 'Adam':
                        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
                    else:
                        optimizer = torch.optim.SGD(cnn.parameters(), lr=lr)

                    train(train_path,cnn,loss_func,optimizer)
                    test_loss,accuracy = test(test_path,cnn,loss_func)
                    print('learning_rate:' ,lr,'| optimizer: ',op,'| dropout: ',do,'| test_loss: ',test_loss)

        if test_loss > best_loss:
            iteration += 1
            print('Loss was not improved, iteration {0}'.format(str(iteration)))
        else:
            print('Saving model...')
            iteration = 0
            best_loss = test_loss
            state = {
                'net': cnn,
                'acc': test_loss,
                'epoch': epoch,
            }

            torch.save(state, 'model/model_cnn.pkl')
            epoch += 1
