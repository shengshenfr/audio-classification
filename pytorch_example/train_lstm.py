import sys
import os
import glob

import numpy as np
import math
import time
from util import split_data,evaluate
from extration import *

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn import metrics

# import multiprocessing
# import GPUtil as GPU


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


def train_rnn(train_features,train_labels,prediction_features,prediction_labels,model,optimizer,loss_func,input_size,batch_size,epochs,split_ratio):

    train_x,train_y,test_x,test_y = split_data(train_features,train_labels,split_ratio)
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


    prediction_features1, prediction_labels1 = torch.from_numpy(prediction_features).float(), torch.from_numpy(prediction_labels).long()
    prediction_features = Variable(prediction_features1, requires_grad=False).type(torch.FloatTensor)
    prediction_labels = prediction_labels1.numpy().squeeze() # covert to numpy array

    # p = multiprocessing.Process(target = monitor_gpu)
    # p.start()
    start_time = time.time()
    for epoch in range(epochs):
        global_epoch_loss = 0
        for step, (x, y) in enumerate(train_loader):   # gives batch data
            #print (x.shape)
            b_x = Variable(x.view(-1,1,input_size), requires_grad=False)

            #print (b_x.shape)
            #print b_x
            b_y = Variable(y.view(-1), requires_grad=False)   # batch y
            # print (b_y.shape)
            # output = net(b_x)               # rnn output
            output = model(b_x)
            # print(output.shape)
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            global_epoch_loss += loss.data[0]

            if step % 20 == 0:
                test_output = model(test_x.view(-1,1,input_size))

                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

                accuracy = sum(pred_y == test_y) / float(test_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

    end_time = time.time()
    training_time = end_time-start_time
    # p.terminate()
    pre_output = model(prediction_features.view(-1,1,input_size))

    pred_y = torch.max(pre_output, 1)[1].data.numpy().squeeze()
    # accuracy = sum(pred_y == test_y) / float(test_y.size)

    accuracy,precision,recall,f1,auc = evaluate(prediction_labels,pred_y)
    # print float(global_epoch_loss.numpy())
    # print train_x.size
    # print train_x.shape
    loss = float(global_epoch_loss.numpy())/float(train_x.size)
    loss = "{:.4f}".format(loss)
    # print loss
    training_time = "{:.4f} s".format(training_time)

    return loss,model,accuracy,precision,recall,f1,auc,training_time
