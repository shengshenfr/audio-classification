import sys
import os
import glob

import numpy as np
import math

import util
from extration import *

from sklearn import preprocessing,metrics

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F


hidden_size = 80
num_layers = 2
num_classes = 3
batch_size = 50
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



def train_rnn(features_normalisation,labels_encode,learning_rate,optimizer,drop_out,type):

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

    if type =='mfcc':
        input_size = 13
    elif type =='wavelet':
        input_size = 12
    elif type == 'rawSignal':
        input_size =200

    rnn = RNN(input_size, hidden_size, num_layers, num_classes,drop_out)
    # print(rnn)

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()


    for epoch in range(num_epochs):
        global_epoch_loss = 0
        for step, (x, y) in enumerate(train_loader):   # gives batch data
            #print (x.shape)
            if type =='mfcc':
                b_x = Variable(x.view(-1,1,13), requires_grad=False)
            elif type =='wavelet':
                b_x = Variable(x.view(-1,1,12), requires_grad=False)
            elif type == 'rawSignal':
                b_x = Variable(x.view(-1,1,200), requires_grad=False)
            #print (b_x.shape)
            #print b_x
            b_y = Variable(y.view(-1), requires_grad=False)   # batch y
            # print (b_y)
            # output = net(b_x)               # rnn output
            output = rnn(b_x)
            # print output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            global_epoch_loss += loss.data[0]

            if step % 50 == 0:
                if type =='mfcc':
                    test_output = rnn(test_x.view(-1,1,13))
                elif type =='wavelet':
                    test_output = rnn(test_x.view(-1,1,12))
                elif type == 'rawSignal':
                    test_output = rnn(test_x.view(-1,1,200))

                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

                accuracy = sum(pred_y == test_y) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

    if type =='mfcc':
        pre_output = rnn(test_x.view(-1,1,13))
    elif type =='wavelet':
        pre_output = rnn(test_x.view(-1,1,12))
    elif type == 'rawSignal':
        pre_output = rnn(test_x.view(-1,1,200))
    pred_y = torch.max(pre_output, 1)[1].data.numpy().squeeze()
    # accuracy = sum(pred_y == test_y) / float(test_y.size)
    # print (metrics.classification_report(test_y, pred_y))

    # print float(global_epoch_loss.numpy())
    # print float(test_y.size)
    return float(global_epoch_loss.numpy())/float(test_y.size),rnn



if __name__ == "__main__":


    # cmd = "rm -rf model/*"
    # sh.run(cmd)
    '''
    learning_rate = 0.01
    optimizer = 'Adam'
    dropout = 0.05
    types = ['mfcc','wavelet','rawSignal']
    features_mfcc = np.loadtxt("feature/train_features_mfcc.txt")
    labels_mfcc = np.loadtxt("feature/train_label_mfcc.txt")
    # print np.unique(labels_mfcc)

    features_wavelet = np.loadtxt("feature/train_features_wavelet.txt")
    labels_wavelet = np.loadtxt("feature/train_label_wavelet.txt")
    # print np.unique(labels_wavelet)

    features_rawSignal = np.loadtxt("feature/train_features_rawSignal.txt")
    labels_rawSignal = np.loadtxt("feature/train_label_rawSignal.txt")
    # print np.unique(labels_rawSignal)

    for type in types:
        print(type)
        if type =='mfcc':
            _,rnn1 = train_rnn(features_mfcc,labels_mfcc,learning_rate,optimizer,dropout,type)
            torch.save(rnn1, 'model/mfcc_model_lstm.pkl')
        elif type == 'wavelet':
            _,rnn2 = train_rnn(features_wavelet,labels_wavelet,learning_rate,optimizer,dropout,type)
            torch.save(rnn2, 'model/wavelet_model_lstm.pkl')
        elif type == 'rawSignal':
            _,rnn3 = train_rnn(features_rawSignal,labels_rawSignal,learning_rate,optimizer,dropout,type)
            torch.save(rnn3, 'model/rawSignal_model_lstm.pkl')

    '''
###########################   choose best modele


    type = 'mfcc'
    features_mfcc = np.loadtxt("feature/train_features_mfcc.txt")
    labels_mfcc = np.loadtxt("feature/train_label_mfcc.txt")
    learning_rate = [0.01,0.1,0.5]
    opt = ['Adam','SGD']
    drop_out = [0.05,0.1,0.2]

    best_loss = np.inf

    for lr in learning_rate:
        for op in opt:
            for do in drop_out:
                loss,rnn = train_rnn(features_mfcc,labels_mfcc,lr,op,do,type)
                print('learning_rate:' ,lr,'| optimizer: ',op,'| dropout: ',do,'| loss: ',loss)

                if loss > best_loss:
                    print('loss was not improved')
                else:
                    print('Saving model...')
                    best_loss = loss
                    torch.save(rnn, 'model/best_mfcc_model_lstm.pkl')
