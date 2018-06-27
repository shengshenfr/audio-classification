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
import librosa
from sklearn import preprocessing,metrics

from extration import parse_audio_files_rawSignal,extration_rawSignal

LAYER_SIZE = 4  # 10 in paper
STACK_SIZE = 2  # 5 in paper
IN_CHANNELS = 5  # 256 in paper. quantized and one-hot input.
RES_CHANNELS = 20 # 512 in paper
EPOCHS = 1
BATCH_SIZE =1
LEARNING_RATE = 0.01


def wavenet():
    net = WaveNet(LAYER_SIZE, STACK_SIZE, IN_CHANNELS, RES_CHANNELS)

    print(net)
    return net

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

def parse_audio_files_mfcc(max_len,read_dir,sub_read,file_ext,rec_fields):

    features, labels = np.empty((0,max_len)), np.empty(0)
    for label, sub_dir in enumerate(sub_read):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(read_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]
            try:
                X, sample_rate = librosa.load(f)
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=max_len).T,axis=0)
                # print len(mfccs),len(chroma),len(mel),len(contrast)
                print (len(mfccs))
                #print ("mfcc is",np.array(mfccs))
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue

            # ext_features = np.hstack([mfccs,chroma,mel,contrast])
            ext_features = np.hstack([mfccs])
            #print len(ext_features)
            #print ext_features
            features = np.vstack([features,ext_features])
            # labels = np.append(labels, quality)
            labels = np.append(labels, label)
        print (features.shape)
        print (features)
        #print labels
    return np.array(features), np.array(labels, dtype = np.int),max_len



def one_hot_encode_label(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    #print labels_encode
    one_hot_encode[np.arange(n_labels), labels] = 1

    return one_hot_encode



def encode_label(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    labels_encode = np.zeros((n_labels,1))
    #print labels_encode
    for i in range(n_labels):
        labels_encode[i]= labels[i]

    return labels_encode


def normaliser_features(features):

    features_normalisation = preprocessing.scale(features)

    return features_normalisation

def train_wavenet_mfcc(net,rec_fields,features,labels):

    train_x,train_y,test_x,test_y = split_data(features,labels)
    x, y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    train_dataset = Data.TensorDataset(x, y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)


    test_x1, test_y1 = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()
    test_x = Variable(test_x1, requires_grad=False).type(torch.FloatTensor)
    test_y = test_y1.numpy().squeeze() # covert to numpy array


    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        # global_epoch_loss = 0
        for step, (x, y) in enumerate(train_loader):   # gives batch data
            # print x.shape
            b_x = Variable(x.view(-1,IN_CHANNELS,rec_fields+1), requires_grad=False)

            b_y = Variable(y.view(-1), requires_grad=False)   # batch y
            output = net(b_x)
            # print b_y.reshape(1,-1).shape
            # print b_y.reshape(1,-1)
            loss = loss_func(output, b_y.reshape(1,-1))   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            # global_epoch_loss += loss.data[0]

            if step % 50 == 0:
                test_output = net(test_x.view(-1,IN_CHANNELS,rec_fields+1))
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size)
                print('train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)


    pre_output = net(test_x.view(-1,IN_CHANNELS,rec_fields+1))
    pred_y = torch.max(pre_output, 1)[1].data.numpy().squeeze()
    # accuracy = sum(pred_y == test_y) / float(test_y.size)
    print (metrics.classification_report(test_y, pred_y))

def train_wavenet_rawSignal(net,rec_fields,features,labels):
    train_x,train_y,test_x,test_y = split_data(features,labels)
    x, y = torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    train_dataset = Data.TensorDataset(x, y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)


    test_x1, test_y1 = torch.from_numpy(test_x).float(), torch.from_numpy(test_y).long()
    test_x = Variable(test_x1, requires_grad=False).type(torch.FloatTensor)
    test_y = test_y1.numpy().squeeze() # covert to numpy array

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            # print x.shape
            b_x = Variable(x.view(-1,IN_CHANNELS,rec_fields+1), requires_grad=False)

            b_y = Variable(y.view(-1), requires_grad=False)   # batch y
            output = net(b_x)
            loss = loss_func(output, b_y.reshape(1,-1))   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if step % 50 == 0:
                test_output = net(test_x.view(-1,IN_CHANNELS,rec_fields+1))
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size)
                print('train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

    pre_output = net(test_x.view(-1,IN_CHANNELS,rec_fields+1))
    pred_y = torch.max(pre_output, 1)[1].data.numpy().squeeze()
    # accuracy = sum(pred_y == test_y) / float(test_y.size)
    print (metrics.classification_report(test_y, pred_y))

if __name__ == '__main__':

    net = wavenet()
    rec_fields = net.receptive_fields
    print ("receptive_fields is ",rec_fields)
    read_dir = "read"
    sub_read = ['Bm','Eg']
    file_ext = '*.wav'
    '''
    max_len = (rec_fields + 1)*IN_CHANNELS
    print max_len
    features_mfcc,labels_mfcc,max_len = parse_audio_files_mfcc(max_len,read_dir,sub_read,file_ext, net.receptive_fields)
    features_mfcc = normaliser_features(features_mfcc)
    # labels_mfcc = one_hot_encode_label(labels_mfcc)
    labels_mfcc = encode_label(labels_mfcc)

    np.savetxt("feature/wavenet_feature_mfcc.txt",features_mfcc)
    np.savetxt("feature/wavenet_label_mfcc.txt",labels_mfcc)



    features = np.loadtxt("feature/wavenet_feature_mfcc.txt")
    labels = np.loadtxt("feature/wavenet_label_mfcc.txt")
    # print features.shape
    train_wavenet_mfcc(net,rec_fields,features,labels)

    '''
    sample_rate = 160
    sample_size = (rec_fields + 1)*IN_CHANNELS

    wavenet_features_rawSignal,wavenet_labels_rawSignal = parse_audio_files_rawSignal(read_dir,sub_read,file_ext,sample_size,sample_rate)
    wavenet_features_rawSignal = normaliser_features(wavenet_features_rawSignal)
    wavenet_labels_rawSignal = encode_label(wavenet_labels_rawSignal)
    np.savetxt("feature/wavenet_features_rawSignal.txt",wavenet_features_rawSignal)
    np.savetxt("feature/wavenet_labels_rawSignal.txt",wavenet_labels_rawSignal)



    wavenet_features_rawSignal = np.loadtxt("feature/wavenet_features_rawSignal.txt")
    wavenet_labels_rawSignal = np.loadtxt("feature/wavenet_labels_rawSignal.txt")
    print wavenet_features_rawSignal.shape,wavenet_labels_rawSignal.shape
    train_wavenet_rawSignal(net,rec_fields,wavenet_features_rawSignal,wavenet_labels_rawSignal)
