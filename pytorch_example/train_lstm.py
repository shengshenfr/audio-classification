
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


input_size = 13
# input_size = 12

hidden_size = 80
num_layers = 2
num_classes = 3
batch_size = 1
num_epochs = 1



class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes,drop_out):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=drop_out)

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



def train_rnn(features_normalisation,labels_encode,learning_rate,optimizer,drop_out):

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

    rnn = RNN(input_size, hidden_size, num_layers, num_classes,drop_out)
    # print(rnn)

    if op == 'Adam':
        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    acc = []

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
            output = rnn(b_x)
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if step % 50 == 0:
                # test_output = net(test_x)
                # test_output = rnn(test_x.view(-1,1,13))
                test_output = rnn(test_x.view(-1,1,13))
                # test_output = rnn(test_x.view(-1,1,12))
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size)
                acc.append(accuracy)
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

    # model_path = 'model/rnn.pkl'
    model_path = 'model/rnn.pkl'
    # model_path = 'model/rnn_wavelet.pkl'
    torch.save(rnn,model_path)
    return max(acc)



if __name__ == "__main__":
    cmd = "rm -rf model/*"
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
    learning_rate = [0.01,0.1,0.5]
    opt = ['Adam','SGD']
    drop_out = [0.05,0.1,0.2]
    for lr in learning_rate:
        for op in opt:
            for do in drop_out:
                accuracy = train_rnn(features_normalisation,labels_encode,lr,op,do)
                print('learning_rate:' ,lr,'| optimizer: ',op,'| dropout: ',do,'| accuracy: ',accuracy)
