
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from sklearn import preprocessing,metrics
import sys
import os
import glob
import time

import numpy as np

import librosa
import librosa.display

import matplotlib.pyplot as plt
import pylab

import sh




batch_size = 5
num_epochs = 1
num_classes = 3
num_kernel_size = 2

class CNN(nn.Module):
    def __init__(self,drop_out,totalNumOfFeatures,max_len,in_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                # in_channels=1,              # input height
                in_channels=in_channels,
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=num_kernel_size),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.Dropout2d(p = drop_out),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=num_kernel_size),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * (totalNumOfFeatures/(num_kernel_size*num_kernel_size)) * (totalNumOfFeatures/(num_kernel_size*num_kernel_size)), num_classes)   # fully connected layer, output 2 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization




def train_mfcc(train_features,train_labels,cnn,loss_func,optimizer):
    print('Loading data...')
    start_time = time.time()
    x, y = torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).long()
    train_dataset = Data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=20)

    for epoch in range(num_epochs):
        cnn.train()
        global_epoch_loss = 0
        for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            # print x.shape
            # print x
            # print y.shape
            b_x = Variable(torch.unsqueeze(x, dim=1))   # batch x
            b_y = Variable(y)   # batch y
            # print b_x.shape
            # print b_y.shape
            output = cnn(b_x)[0]               # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            global_epoch_loss += loss.data[0]
            if step % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, step * len(b_x), len(train_loader.dataset), 100.* step / len(train_loader), loss.data[0]))

    return global_epoch_loss / len(train_loader.dataset)




def test_mfcc(test_features,test_labels,cnn):

    test_x, test_y = torch.from_numpy(test_features).float(), torch.from_numpy(test_labels).long()
    test_x = Variable(torch.unsqueeze(test_x, dim=1), requires_grad=False)

    test_output, last_layer = cnn(test_x)
    # print test_output
    pred_y = torch.max(test_output, 1)[1].data.squeeze()
    # print test_y.size(0)
    # print float(sum(pred_y == test_y))
    accuracy = sum(pred_y.numpy() == test_y.numpy()) / float(test_y.size(0))

    # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

    # print type(loss.data[0].numpy())
    # pre_output,_ = cnn(test_x)
    # pred_y = torch.max(pre_output, 1)[1].data.numpy().squeeze()
    print (metrics.classification_report(test_y.numpy(), pred_y))

    return accuracy



def train_rawSignal(image_train_path,cnn,loss_func,optimizer,totalNumOfFeatures,max_len):
    transform = transforms.Compose(
        [transforms.Scale([totalNumOfFeatures,max_len]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.ImageFolder(root=image_train_path, transform=transform )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=2)

    for epoch in range(num_epochs):
        cnn.train()
        global_epoch_loss = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            # print inputs.shape
            # print labels
            inputs, labels = Variable(inputs), Variable(labels)
            output = cnn(inputs)[0]               # cnn output
            # print output
            loss = loss_func(output, labels)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            global_epoch_loss += loss.data[0]
            if i % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(train_loader.dataset), 100.* i / len(train_loader), loss.data[0]))
    return global_epoch_loss / len(train_loader.dataset)


def test_rawSignal(image_test_path,cnn,totalNumOfFeatures,max_len):
    transform = transforms.Compose(
        [transforms.Scale([totalNumOfFeatures,max_len]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_data = torchvision.datasets.ImageFolder(root=image_test_path, transform=transform )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=2)
    # print test_data
    # dataiter = iter(test_loader)
    # images, labels = dataiter.next()
    # print images.shape,labels.shape
    cnn.eval()
    correct = 0
    pred_y = np.array([])
    test_y = np.array([])
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        # print inputs.reshape
        # print labels.shape
        output = cnn(inputs)
        # print type(output)
        # print output[0]
        pred = output[0].max(1, keepdim=True)[1]
        pred_y = np.append(pred_y,pred)
        test_y = np.append(test_y,labels)
        # print pred_y
        # pred = torch.max(output[0], 1)[1].data.squeeze()
        # print pred
        # print labels.data
        # print pred.eq(labels.data.view_as(pred)).sum()
        correct += pred.eq(labels.data.view_as(pred)).sum()
    # print correct.numpy()
    # print len(test_loader.dataset)
    # print float(correct) /len(test_loader.dataset)
    print (metrics.classification_report(test_y, pred_y))
    return float(correct) /len(test_loader.dataset)







if __name__ == "__main__":
    cmd = "rm -rf model/mfcc_model_cnn.pkl"
    sh.run(cmd)



    subs = ['Ba','Bm','Eg']
    file_ext = '*.wav'
    image_ext = '*.png'
    totalNumOfFeatures = 28
    max_len = 28





    '''
    types = ['mfcc','rawSignal']
    # types = ['rawSignal']
    # types = ['mfcc']
    loss_func = nn.CrossEntropyLoss()
    lr = 0.01

    for type in types:
        if type == 'mfcc':
            in_channels = 1
            cnn = CNN(0.1,totalNumOfFeatures,max_len,in_channels)
            print cnn
            optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
            train_loss = train_mfcc(train_features,train_labels,cnn,loss_func,optimizer)
            accuracy = test_mfcc(test_features,test_labels,cnn)

        elif type == 'rawSignal':
            in_channels = 3
            cnn = CNN(0.1,totalNumOfFeatures,max_len,in_channels)
            print cnn
            optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
            train_loss = train_rawSignal(image_train_path,cnn,loss_func,optimizer,totalNumOfFeatures,max_len)
            accuracy = test_rawSignal(image_test_path,cnn,totalNumOfFeatures,max_len)


    '''


    ### use mfcc
    best_loss = np.inf
    learning_rate = [0.01,0.1,0.5]
    opt = ['Adam','SGD']
    drop_out = [0.05,0.1,0.2]
    in_channels = 1

    for do in drop_out:
        cnn = CNN(do,totalNumOfFeatures,max_len,in_channels)
        loss_func = nn.CrossEntropyLoss()
        for lr in learning_rate:
            for op in opt:
                if op == 'Adam':
                    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
                else:
                    optimizer = torch.optim.SGD(cnn.parameters(), lr=lr)

                loss = train_mfcc(train_features,train_labels,cnn,loss_func,optimizer)
                accuracy = test_mfcc(test_features,test_labels,cnn)
                print('learning_rate:' ,lr,'| optimizer: ',op,'| dropout: ',do,'| test_loss: ',float(loss.numpy()),'| accuracy: ',accuracy)

                if loss > best_loss:
                    print('Loss was not improved')
                else:
                    print('Saving model...')
                    best_loss = loss
                    torch.save(cnn, 'model/best_mfcc_model_cnn.pkl')
