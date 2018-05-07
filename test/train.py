import numpy as np
from sys import argv
import sys
import os
import glob
import sh
import matplotlib.pyplot as plt
import util
import math
import torch
from torch import nn
from torch.autograd import Variable
from extration import *
from sklearn import preprocessing
from model import *
import torch.utils.data as Data




def split_data(train_features,train_labels):
    # random train and test sets.
    # train_test_split(features_normalization, lable[0], test_size = 0.2, random_state=0)
    print np.random.rand(len(train_features))
    train_test_split = np.random.rand(len(train_features)) < 0.70
    print train_test_split
    train_x = train_features[train_test_split]
    train_y = train_labels[train_test_split]
    test_x = train_features[~train_test_split]
    test_y = train_labels[~train_test_split]

    print train_x.shape,train_y.shape
    return train_x,train_y,test_x,test_y

def normaliser_features(features):

    features_normalisation = preprocessing.scale(features)

    return features_normalisation



def encode_label(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    labels_encode = np.zeros((n_labels,1))
    #print labels_encode
    for i in range(n_labels):
        labels_encode[i]= labels[i]

    return labels_encode


def train(features,labels):
    x = np.linspace(-7, 10, 2000)[:, np.newaxis]
    print ("x is", x)
    y = np.square(x) - 5
    print ("y is", y)
    print ("features are ",features)
    features_normalisation = normaliser_features(features)
    print ("features noramallisation are ",features_normalisation)

    labels_encode = encode_label(labels)
    print ("label encode is ",labels_encode)

    train_x,train_y,test_x,test_y = split_data(features_normalisation,labels_encode)
    rnn = RNN()
    print(rnn)

    LR = 0.01
    EPOCH = 1

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

    #print("train x is ",train_x)
    #print("train y is ",train_y)
    #
    # x, y = Variable(torch.Tensor(train_x)), Variable(torch.Tensor(train_y))
    #
    # # training and testing
    # for epoch in range(EPOCH):
    #     output = rnn(x)               # rnn output
    #     loss = loss_func(output, y)   # cross entropy loss
    #     optimizer.zero_grad()           # clear gradients for this training step
    #     loss.backward()                 # backpropagation, compute gradients
    #     optimizer.step()                # apply gradients
    #
    #     print(loss.data[0])




if __name__ == "__main__":

    result_redimension_dir = "result_redimension"
    sub_redimensions = ['bm_redimension', 'eg_redimension']
    file_ext='*.wav'
    features,labels = parse_audio_files_librosa(result_redimension_dir,sub_redimensions,file_ext)
    train(features,labels)
