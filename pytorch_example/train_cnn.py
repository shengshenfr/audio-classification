
import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn import preprocessing,metrics
import sys
import os
import glob
import sh
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pylab
import librosa.display


import torchvision
import torchvision.transforms as transforms




batch_size = 1
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





def train_mfcc(train_features,train_labels,test_features,test_labels,cnn,loss_func,optimizer):
    print('Loading data...')
    start_time = time.time()
    x, y = torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).long()
    train_dataset = Data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=20)

    test_x, test_y = torch.from_numpy(test_features).float(), torch.from_numpy(test_labels).long()
    test_x = Variable(torch.unsqueeze(test_x, dim=1), requires_grad=False)

    test_loss = np.empty(0)
    acc = []
    for epoch in range(num_epochs):
        for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            # print x.shape
            # print x
            # print y.shape
            b_x = Variable(torch.unsqueeze(x, dim=1))   # batch x
            b_y = Variable(y)   # batch y
            print b_x.shape
            print b_y.shape
            output = cnn(b_x)[0]               # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            if step % 50 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                # print test_y.size(0)
                # print float(sum(pred_y == test_y))
                accuracy = sum(pred_y.numpy() == test_y.numpy()) / float(test_y.size(0))

                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

                # print type(loss.data[0].numpy())
                test_loss = np.append(test_loss, loss.data[0].numpy())
                acc.append(accuracy)
    pre_output,_ = cnn(test_x)
    pred_y = torch.max(pre_output, 1)[1].data.numpy().squeeze()
    print (metrics.classification_report(test_y.numpy(), pred_y))

    return min(test_loss),max(acc)



def get_mfccs(redimension_dir,redimension_subs,file_ext,totalNumOfFeatures,max_len):

    features = np.empty((0,totalNumOfFeatures,max_len))

    labels = np.empty(0)
    for label, sub_dir in enumerate(redimension_subs):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(redimension_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]

            try:
                X, sample_rate = librosa.load(f)
                mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=totalNumOfFeatures)
                print mfccs.shape
                print ("mfcc is",mfccs)
                if mfccs.shape[1] < max_len:
                    pad = np.zeros((mfccs.shape[0], max_len - mfccs.shape[1]))
                    mfccs = np.hstack((mfccs, pad))
                elif mfccs.shape[1] > max_len:
                    mfccs = mfccs[:,:max_len ]

                mfccs = torch.FloatTensor(mfccs)
                mean = mfccs.mean()
                std = mfccs.std()
                if std != 0:
                    mfccs.add_(-mean)
                    mfccs.div_(std)
                #print ("mfcc is",np.array(mfccs))
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue

            ext_features = np.hstack([mfccs])
            #print len(ext_features)
            print ext_features.shape
            ext_features = np.resize(ext_features, (1, ext_features.shape[0],ext_features.shape[1]))
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)
        print (features.shape)
        # print (features)
        #print labels
    return np.array(features), np.array(labels, dtype = np.int)

def train_rawSignal(image_train_path,image_test_path,cnn,loss_func,optimizer,totalNumOfFeatures,max_len):
    transform = transforms.Compose(
        [transforms.Scale([totalNumOfFeatures,max_len]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.ImageFolder(root=image_train_path, transform=transform )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=2)

    test_data = torchvision.datasets.ImageFolder(root=image_test_path, transform=transform )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=2)
    print test_data
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    print images,labels
    #
    # for epoch in range(num_epochs):
    #
    #     for i, data in enumerate(train_loader, 0):
    #         # get the inputs
    #         inputs, labels = data
    #         # print inputs.shape
    #         # print labels
    #         inputs, labels = Variable(inputs), Variable(labels)
    #         output = cnn(inputs)[0]               # cnn output
    #         # print output
    #         loss = loss_func(output, labels)   # cross entropy loss
    #         optimizer.zero_grad()           # clear gradients for this training step
    #         loss.backward()                 # backpropagation, compute gradients
    #         optimizer.step()                # apply gradients


            # if i % 50 == 0:
            #     test_output, last_layer = cnn(test_x)
            #     pred_y = torch.max(test_output, 1)[1].data.squeeze()
            #     # print test_y.size(0)
            #     # print float(sum(pred_y == test_y))
            #     accuracy = sum(pred_y.numpy() == test_y.numpy()) / float(test_y.size(0))
            #
            #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)
            #
            #     # print type(loss.data[0].numpy())
            #     test_loss = np.append(test_loss, loss.data[0].numpy())
            #     acc.append(accuracy)


def rawSignal_to_image(redimension_dir,redimension_subs,file_ext,image_dir):
    for label, sub_dir in enumerate(redimension_subs):
        # print("label: %s" % (label))
        # print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(redimension_dir, sub_dir, file_ext)):
            # print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[3]
            spice = (os.path.splitext(f)[0]).split(os.sep)[2]
            print waveFile_name
            sig, fs = librosa.load(f)


            pylab.axis('off') # no axis
            pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
            S = librosa.feature.melspectrogram(y=sig, sr=fs)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))



            # make pictures name
            save_path = image_dir+ "/" + spice + "/" + waveFile_name + '.png'
            pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
            pylab.close()

def clean(file_dir):
    for i, f in enumerate(glob.glob(file_dir + os.sep +'*')):
        # print f
        cmd = "rm -rf " + f  + "/*.png"
        sh.run(cmd)


if __name__ == "__main__":
    cmd = "rm -rf model/model_cnn.pkl"
    sh.run(cmd)

    train_path = "redimension/train"
    test_path = "redimension/test"

    image_train_path = "image/train"
    image_test_path = "image/test"

    subs = ['Ba','Bm','Eg']
    file_ext = '*.wav'
    image_ext = '*.png'
    totalNumOfFeatures = 28
    max_len = 28

    # clean(image_train_path)
    # clean(image_test_path)

    # train_features,train_labels = get_mfccs(train_path,subs,file_ext,totalNumOfFeatures,max_len)
    # test_features,test_labels = get_mfccs(test_path,subs,file_ext,totalNumOfFeatures,max_len)

    # rawSignal_to_image(train_path,subs,file_ext,image_train_path)
    # rawSignal_to_image(test_path,subs,file_ext,image_test_path)


    # types = ['mfcc','rawSignal']
    types = ['rawSignal']
    loss_func = nn.CrossEntropyLoss()
    lr = 0.01

    for type in types:
        if type == 'mfcc':
            in_channels = 1
            cnn = CNN(0.1,totalNumOfFeatures,max_len,in_channels)
            print cnn
            optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
            train_mfcc(train_features,train_labels,test_features,test_labels, cnn,loss_func,optimizer)

        elif type == 'rawSignal':
            in_channels = 3
            cnn = CNN(0.1,totalNumOfFeatures,max_len,in_channels)
            print cnn
            optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
            train_rawSignal(image_train_path,image_test_path,cnn,loss_func,optimizer,totalNumOfFeatures,max_len)



    '''
    best_loss = np.inf
    learning_rate = [0.01,0.1,0.5]
    opt = ['Adam','SGD']
    drop_out = [0.05,0.1,0.2]

    for do in drop_out:
        cnn = CNN(do,totalNumOfFeatures,max_len)
        loss_func = nn.CrossEntropyLoss()
        for lr in learning_rate:
            for op in opt:
                if op == 'Adam':
                    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
                else:
                    optimizer = torch.optim.SGD(cnn.parameters(), lr=lr)

                test_loss,acc = train(train_features,train_labels,test_features,test_labels, cnn,loss_func,optimizer)
                print('learning_rate:' ,lr,'| optimizer: ',op,'| dropout: ',do,'| test_loss: ',test_loss,'| accuracy: ',acc)

                if test_loss > best_loss:
                    print('Loss was not improved')
                else:
                    print('Saving model...')
                    best_loss = test_loss
                    torch.save(cnn, 'model/model_cnn.pkl')
    '''
