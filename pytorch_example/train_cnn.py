
import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F

import sys
import os
import glob
import sh
import numpy as np
import librosa


batch_size = 1
num_epochs = 1

class CNN(nn.Module):
    def __init__(self,drop_out):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.Dropout2d(p = drop_out),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 3)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization





def train(train_features,train_labels,test_features,test_labels,cnn,loss_func,optimizer):
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
    return min(test_loss),max(acc)



def get_features(redimension_dir,redimension_subs,file_ext):
    totalNumOfFeatures = 28
    max_len = 28
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



if __name__ == "__main__":
    cmd = "rm -rf model/model_cnn.pkl"
    sh.run(cmd)

    train_path = "redimension/train"

    test_path = "redimension/test"

    redimension_subs = ['Ba', 'Bm','Eg']
    file_ext='*.wav'
    train_features,train_labels = get_features(train_path,redimension_subs,file_ext)

    test_features,test_labels = get_features(test_path,redimension_subs,file_ext)
    '''
    cnn = CNN(0.1)
    print cnn
    loss_func = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    train(train_features,train_labels,test_features,test_labels, cnn,loss_func,optimizer)
    '''
    best_loss = np.inf
    learning_rate = [0.01,0.1,0.5]
    opt = ['Adam','SGD']
    drop_out = [0.05,0.1,0.2]

    for do in drop_out:
        cnn = CNN(do)
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
