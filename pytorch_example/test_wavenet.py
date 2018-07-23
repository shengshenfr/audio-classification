from wavenet_example import WaveNet
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F

import numpy as np
import sys
import os
import glob

from sklearn import preprocessing,metrics
import util
import time


LAYER_SIZE = 4  # 10 in paper
STACK_SIZE = 2  # 5 in paper
IN_CHANNELS = 5  # 256 in paper. quantized and one-hot input.
RES_CHANNELS = 20 # 512 in paper



def wavenet():
    net = WaveNet(LAYER_SIZE, STACK_SIZE, IN_CHANNELS, RES_CHANNELS)

    rec_fields = net.receptive_fields
    print ("receptive_fields is ",rec_fields)
    max_size = (rec_fields + 1)*IN_CHANNELS
    print(net)
    return net,rec_fields,max_size


def train_wavenet(net,rec_fields,train_features,train_labels,validation_features,validation_labels,
                                        optimizer,loss_func,batch_size,epochs):

    # train_x,train_y,test_x,test_y = util.split_data(train_features,train_labels,split_ratio)
    x, y = torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).long()
    train_dataset = Data.TensorDataset(x, y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,)


    # test_x1, test_y1 = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()
    # test_x = Variable(test_x1, requires_grad=False).type(torch.FloatTensor)
    # test_y = test_y1.numpy().squeeze() # covert to numpy array

    validation_features, validation_labels = torch.from_numpy(validation_features).float(), torch.from_numpy(validation_labels).long()
    validation_features = Variable(validation_features, requires_grad=False).type(torch.FloatTensor)
    validation_labels = validation_labels.numpy().squeeze() # covert to numpy array

    start_time = time.time()

    for epoch in range(epochs):
        global_epoch_loss = 0
        for step, (x, y) in enumerate(train_loader):   # gives batch data
            # print x.shape
            b_x = Variable(x.view(-1,IN_CHANNELS,rec_fields+1), requires_grad=False)

            b_y = Variable(y.view(-1), requires_grad=False)   # batch y
            output = net(b_x)
            # print ("b_y",b_y.shape)
            # print ("output",output.shape)
            # print (b_y.reshape(1,-1).shape)
            # print (b_y.reshape(1,-1))
            loss = loss_func(output, b_y.reshape(b_y.shape[0],-1))   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            global_epoch_loss += loss.data[0]

            if step % 50 == 0:
                test_output = net(validation_features.view(-1,IN_CHANNELS,rec_fields+1))
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == validation_labels) / float(validation_labels.size)
                print('train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

    end_time = time.time()
    training_time = end_time-start_time

    pre_output = net(validation_features.view(-1,IN_CHANNELS,rec_fields+1))
    pred_y = torch.max(pre_output, 1)[1].data.numpy().squeeze()
    # accuracy = sum(pred_y == test_y) / float(test_y.size)
    # print (metrics.classification_report(prediction_labels, pred_y))
    accuracy,precision,recall,f1 = util.evaluate(validation_labels,pred_y)
    loss = float(global_epoch_loss.numpy())/float(train_features.size)
    loss = "{:.4f}".format(loss)
    # print (loss)
    training_time = "{:.4f} s".format(training_time)

    return loss,accuracy,precision,recall,f1,training_time
