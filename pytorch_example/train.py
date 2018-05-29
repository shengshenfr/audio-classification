
import sys
import os
import glob

import numpy as np
import math
import matplotlib.pyplot as plt

import util
from extration import *

from sklearn import preprocessing

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
import time
from loader import GCommandLoader

input_size = 13
# input_size = 12

hidden_size = 80
num_layers = 2
num_classes = 3
batch_size = 1
num_epochs = 1
learning_rate = 0.01


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

       # print (out.size())
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

class Net(torch.nn.Module):

    def __init__(self, feature_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(feature_size,hidden_size)
        self.out = torch.nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)

        return x



class CNN(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5,stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1258, 1000)
        self.fc2 = nn.Linear(1000, 30)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def split_data(train_features,train_labels):
    # random train and test sets.
    # train_test_split(features_normalization, lable[0], test_size = 0.2, random_state=0)
    #print np.random.rand(len(train_features))
    train_test_split = np.random.rand(len(train_features)) < 0.70
    #print train_test_split
    train_x = train_features[train_test_split]
    train_y = train_labels[train_test_split]
    test_x = train_features[~train_test_split]
    test_y = train_labels[~train_test_split]

    #print train_x.shape,train_y.shape
    return train_x,train_y,test_x,test_y

def toEachTensor(features_normalisation):
    # features_normalisation = np.resize(features_normalisation, (features_normalisation.shape[0],1, features_normalisation.shape[1]))
    print features_normalisation.shape
    # print features_normalisation
    f = np.array
    for i in range(len(features_normalisation)):
        temp = features_normalisation[i]
        print temp
        print temp.shape
        temp = np.resize(temp, (1, temp.shape[0]))
        temp = np.resize(temp, (1, temp.shape[0],temp.shape[1]))
        t = torch.FloatTensor(temp)
        f.append(t)
        print t.shape
    print len(f)
    return f


def train_cnn(features_normalisation,labels_encode):
    print('Loading data...')
    start_time = time.time()
    train_path = "read/"
    train_dataset = GCommandLoader(train_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
    num_workers=20)
    model = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for batch_idx, (data, target) in enumerate(train_loader):
        print data.shape
        print target.shape
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()


    '''
    print len(features_normalisation)
    # features_normalisation = toEachTensor(features_normalisation)
    train_x,train_y,test_x,test_y = split_data(features_normalisation,labels_encode)
    # train_x = np.resize(train_x, (1, train_x.shape[0], train_x.shape[1]))
    # train_y = np.resize(train_y, (1, train_y.shape[0]))
    # print train_x
    print type(train_x)
    x = torch.FloatTensor(train_x)


    print train_x.shape
    y =  torch.from_numpy(train_y).long()

    print x.shape
    print y.shape
    train_dataset = Data.TensorDataset(x, y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    print('Configuring CNN model...')
    model = CNN()
    print(model)

    # optimizer and loss function
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # set the mode to train
    print("Training and evaluating...")
    best_acc = 0.0
    for epoch in range(num_epochs):
        # load the training data in batch
        model.train()


        for step, (x, y) in enumerate(train_loader):   # gives batch data
            print x.shape
            inputs, targets = Variable(x),Variable(y)
            print inputs.shape
            print targets
            optimizer.zero_grad()
            outputs = model(inputs)  # forward computation
            loss = criterion(outputs, targets)

            # backward propagation and update parameters
            loss.backward()
            optimizer.step()
'''


def train_rnn(features_normalisation,labels_encode):

    train_x,train_y,test_x,test_y = split_data(features_normalisation,labels_encode)
    x, y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    train_dataset = Data.TensorDataset(x, y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,)
    #print("train x is ",train_x)
    #print("train y is ",train_y)
    # x_temp =  train_x[ : , np.newaxis , :]
    # print x_temp.shape
    # x, y = Variable(torch.Tensor(x_temp)), Variable(torch.LongTensor(train_y))

    test_x1, test_y1 = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()
    test_x = Variable(test_x1, requires_grad=False).type(torch.FloatTensor)
    test_y = test_y1.numpy().squeeze() # covert to numpy array

    rnn = RNN(input_size, hidden_size, num_layers, num_classes)
    print(rnn)
    # cnn = CNN()
    # print(cnn)
    # net = Net(feature_size = 13,hidden_size = 10, output_size = 2)
    # print(net)

    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)   # optimize all parameters
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted


    for epoch in range(num_epochs):
        for step, (x, y) in enumerate(train_loader):   # gives batch data
            #print (x.shape)
            # b_x = Variable(x.view(-1,13))

            b_x = Variable(x.view(-1,1,13), requires_grad=False)
            # b_x = Variable(x.view(-1,1,12), requires_grad=False)
            #print (b_x.shape)
            #print b_x
            b_y = Variable(y.view(-1), requires_grad=False)   # batch y
            #print (b_y)
            # output = net(b_x)               # rnn output
            output = cnn(b_x)
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if step % 20 == 0:
                # test_output = net(test_x)
                # test_output = rnn(test_x.view(-1,1,13))
                test_output = cnn(test_x.view(-1,1,13))
                # test_output = rnn(test_x.view(-1,1,12))
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

    # model_path = 'model/rnn.pkl'
    model_path = 'model/cnn.pkl'
    # model_path = 'model/rnn_wavelet.pkl'
    torch.save(rnn,model_path)

if __name__ == "__main__":
    cmd = "rm -rf model/rnn.pkl"
    sh.run(cmd)
    '''
    read_dir = "read"
    sub_read = ['Bm','Eg']
    file_ext='*.wav'
    features,labels = parse_audio_files_librosa(read_dir,sub_read,file_ext)
    features,labels = parse_audio_files_waveletPackets(read_dir,sub_read,file_ext)
    '''
    features_normalisation = np.loadtxt("feature/prediction_features.txt")
    labels_encode = np.loadtxt("feature/prediction_label.txt")

    print features_normalisation.shape
    print (len(features_normalisation))
    # train_rnn(features_normalisation,labels_encode)
    train_cnn(features_normalisation,labels_encode)
