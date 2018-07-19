import torch
import torch.optim as optim
from torch import nn
import numpy as np

import argparse
import os
import glob
import sys

import librosa
import librosa.display

import matplotlib.pyplot as plt
import pylab

import sh
from util import cut,extract

from initial import main


from train_lstm import RNN,train_rnn
from train_cnn import CNN,train_mfcc,predict_mfcc,train_rawSignal,predict_rawSignal
from test_wavenet import wavenet,train_wavenet
#####  Init: create the files
# main()

######cut files
parser = argparse.ArgumentParser(
    description='audio classification')
parser.add_argument('--train_wav_path', default='wavData/train',
                    help='path to the train wav data folder')
parser.add_argument('--train_csv_path', default='csvData/train',
                    help='path to the train csv data folder')
parser.add_argument('--read_path', default='read',
                    help='stock cut wav files')

##### redimension
parser.add_argument('--padding_path', default='padding',
                    help='path to stock padding audio')

parser.add_argument('--redimension_train_path', default='redimension/train',
                    help='path to redimension train wav folder')
parser.add_argument('--redimension_validation_path', default='redimension/validation',
                    help='path to redimension validation wav folder')

parser.add_argument('--split_ratio', type=float, default=0.7,
                    metavar='N', help='split train/test ratio')
 #############  extract
parser.add_argument('--prepare', default=False,
                    help='if you excute the code at first time. please cut the raw wav and extract features by input True')

parser.add_argument('--image_train_path', default='image/train',
                    help='path to stock train image folder')
parser.add_argument('--image_validation_path', default='image/validation',
                    help='path to stock validation image folder')
parser.add_argument('--mfcc_length', type=int, default=13,
                    metavar='N', help='define the max length of mfcc for lstm')
parser.add_argument('--sample_size', type=int, default=200,
                    metavar='N', help='define the sample size of raw signal for lstm')
parser.add_argument('--sample_rate', type=int, default=100,
                    metavar='N', help='define the sample rate of raw signal for lstm')
parser.add_argument('--length', type=int, default=28,
                    metavar='N', help='define the max length (mfcc or image of raw signal) for cnn')
parser.add_argument('--width', type=int, default=28,
                    metavar='N', help='define the max width (mfcc or image of raw signal) for cnn')

##########  define parameters of network
parser.add_argument('--features_type', default='mfcc',
                    help='type of features: mfcc,wavelet(only for network lstm),raw_signal')
parser.add_argument('--batch_size', type=int, default=1,
                    metavar='N', help='training and valid batch size')

parser.add_argument('--arc', default='lstm',
                    help='network architecture: lstm,cnn,wavenet')

parser.add_argument('--epochs', type=int, default=1,
                    metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam',
                    help='optimization method: sgd | adam')
parser.add_argument('--drop_out', type=float, default=0.1,
                    metavar='N', help='dropout')




parser.add_argument('--num_layers', type=int, default=2,
                    metavar='N', help='number of layers')
parser.add_argument('--hidden_size', type=int, default=80,
                    metavar='N', help='number of hidden size')
parser.add_argument('--num_classes', type=int, default=3,
                    metavar='N', help='number of classes')



args = parser.parse_args()
# print args.cut
# print args.train_csv_path
# print args.train_wav_path

########################   cut raw signal
if args.prepare:

    cut(args.read_path,args.train_csv_path,args.train_wav_path,
        args.redimension_train_path,args.redimension_validation_path,args.padding_path,args.split_ratio)

# if args.extract:

    extract(args.read_path,args.redimension_train_path,args.redimension_validation_path,args.mfcc_length,
                        args.sample_size,args.sample_rate,args.length,args.width,args.image_train_path,args.image_validation_path)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~Loading data

# # types = ['mfcc','wavelet','rawSignal']


if args.features_type == 'mfcc' and args.arc =='lstm':
    # train_features,train_labels = parse_audio_files_mfcc(args.redimension_train_path,sub_dirs,file_ext,args.mfcc_length)
    # prediction_features,prediction_labels = parse_audio_files_mfcc(args.redimension_prediction_path,sub_dirs,file_ext,args.mfcc_length)
    print("begin to train mfcc by lstm")
    train_features = np.loadtxt("feature/lstm_train_features_mfcc.txt")
    train_labels = np.loadtxt("feature/lstm_train_labels_mfcc.txt")
    prediction_features = np.loadtxt("feature/lstm_prediction_features_mfcc.txt")
    prediction_labels = np.loadtxt("feature/lstm_prediction_labels_mfcc.txt")

    input_size = train_features.shape[1]
    # print("input size is ",input_size)
    model = RNN(input_size,args.hidden_size, args.num_layers, args.num_classes,args.drop_out)

    # define optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)

    loss_func = nn.CrossEntropyLoss()


    loss,model,accuracy,precision,recall,f1,auc,training_time = train_rnn(train_features,train_labels,prediction_features,prediction_labels,model,
                                        optimizer,loss_func,input_size,args.batch_size,args.epochs,args.split_ratio)

    torch.save(model, 'model/mfcc_model_lstm.pkl')
    # print type(training_time)
    print ("training time " ,training_time)
    write_result(loss,accuracy,precision,recall,f1,auc,training_time,args.features_type,args.epochs,args.batch_size,args.split_ratio,args.arc)

elif args.features_type == 'wavelet'and args.arc =='lstm':
    # train_features,train_labels = parse_audio_files_waveletPackets(args.redimension_train_path,sub_dirs,file_ext)
    # prediction_features,prediction_labels = parse_audio_files_waveletPackets(args.redimension_prediction_path,sub_dirs,file_ext)
    train_features = np.loadtxt("feature/lstm_train_features_wavelet.txt")
    train_labels = np.loadtxt("feature/lstm_train_labels_wavelet.txt")
    prediction_features = np.loadtxt("feature/lstm_prediction_features_wavelet.txt")
    prediction_labels = np.loadtxt("feature/lstm_prediction_labels_wavelet.txt")

    input_size = train_features.shape[1]
    model = RNN(input_size,args.hidden_size, args.num_layers, args.num_classes,args.drop_out)

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)

    loss_func = nn.CrossEntropyLoss()


    loss,model,accuracy,precision,recall,f1,auc,training_time = train_rnn(train_features,train_labels,prediction_features,prediction_labels,model,
                                        optimizer,loss_func,input_size,args.batch_size,args.epochs,args.split_ratio)

    torch.save(model, 'model/wavelet_model_lstm.pkl')
    print ("training time " ,training_time)
    write_result(loss,accuracy,precision,recall,f1,auc,training_time,args.features_type,args.epochs,args.batch_size,args.split_ratio,args.arc)

elif args.features_type == 'raw_signal'and args.arc =='lstm':
    # train_features,train_labels = parse_audio_files_rawSignal(args.redimension_train_path,sub_dirs,file_ext,args.sample_size,args.sample_rate)
    # prediction_features,prediction_labels = parse_audio_files_rawSignal(args.redimension_train_path,sub_dirs,file_ext,args.sample_size,args.sample_rate
    train_features = np.loadtxt("feature/lstm_train_features_rawSignal.txt")
    train_labels = np.loadtxt("feature/lstm_train_labels_rawSignal.txt")
    prediction_features = np.loadtxt("feature/lstm_prediction_features_rawSignal.txt")
    prediction_labels = np.loadtxt("feature/lstm_prediction_labels_rawSignal.txt")

    input_size = train_features.shape[1]
    # print input_size
    model = RNN(input_size,args.hidden_size, args.num_layers, args.num_classes,args.drop_out)

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)

    loss_func = nn.CrossEntropyLoss()


    loss,model,accuracy,precision,recall,f1,auc,training_time = train_rnn(train_features,train_labels,prediction_features,prediction_labels,model,
                                        optimizer,loss_func,input_size,args.batch_size,args.epochs,args.split_ratio)
    torch.save(model, 'model/rawSignal_model_lstm.pkl')
    print ("training time " ,training_time)
    write_result(loss,accuracy,precision,recall,f1,auc,training_time,args.features_type,args.epochs,args.batch_size,args.split_ratio,args.arc)

elif args.features_type == 'mfcc'and args.arc =='cnn':
    # train_features,train_labels = get_cnn_mfccs(args.redimension_train_path,sub_dirs,file_ext,args.length,args.width)
    # prediction_features,prediction_labels = get_cnn_mfccs(args.redimension_prediction_path,sub_dirs,file_ext,args.length,args.width)
    train_features = np.loadtxt("feature/cnn_train_features_mfcc.txt")
    print train_features.shape[0]
    train_features = train_features.reshape(train_features.shape[0]/args.length,args.length,args.width)
    train_labels = np.loadtxt("feature/cnn_train_labels_mfcc.txt")

    prediction_features = np.loadtxt("feature/cnn_prediction_features_mfcc.txt")
    prediction_features = prediction_features.reshape(prediction_features.shape[0]/args.length,args.length,args.width)
    prediction_labels = np.loadtxt("feature/cnn_prediction_labels_mfcc.txt")


    in_channels = 1
    model = CNN(args.num_classes,args.drop_out,args.length,args.width,in_channels)

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    loss,training_time = train_mfcc(train_features,train_labels,model,
                                        optimizer,loss_func,args.batch_size,args.epochs,args.split_ratio)
    model,accuracy,precision,recall,f1,auc = predict_mfcc(prediction_features,prediction_labels,model)
    torch.save(model, 'model/mfcc_model_cnn.pkl')
    print ("training time " ,training_time)
    write_result(loss,accuracy,precision,recall,f1,auc,training_time,args.features_type,args.epochs,args.batch_size,args.split_ratio,args.arc)

elif args.features_type == 'raw_signal'and args.arc =='cnn':
    # clean_image(args.image_train_path)
    # clean_image(args.image_prediction_path)
    # rawSignal_to_image(args.redimension_train_path,sub_dirs,file_ext,args.image_train_path)
    # rawSignal_to_image(args.redimension_prediction_path,sub_dirs,file_ext,args.image_prediction_path)

    in_channels = 3
    model = CNN(args.num_classes,args.drop_out,args.length,args.width,in_channels)

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    loss,training_time  = train_rawSignal(args.image_train_path,model,
                                        optimizer,loss_func,args.batch_size,args.epochs,args.length,args.width)
    _,model,accuracy,precision,recall,f1,auc = predict_rawSignal(args.image_prediction_path,model,args.batch_size,args.length,args.width)
    torch.save(model, 'model/rawSignal_model_cnn.pkl')
    print ("training time " ,training_time)
    write_result(loss,accuracy,precision,recall,f1,auc,training_time,args.features_type,args.epochs,args.batch_size,args.split_ratio,args.arc)


elif args.features_type == 'mfcc'and args.arc =='wavenet':
    ########### wavenet mfcc
    model,rec_fields,_ = wavenet()

    # wavenet_train_features_mfcc,wavenet_train_labels_mfcc = parse_audio_files_mfcc(args.redimension_train_path,sub_dirs,file_ext,max_size)
    # wavenet_prediction_features_mfcc,wavenet_prediction_labels_mfcc = parse_audio_files_mfcc(args.redimension_prediction_path,sub_dirs,file_ext,max_size)

    train_features = np.loadtxt("feature/wavenet_train_features_mfcc.txt")
    train_labels = np.loadtxt("feature/wavenet_train_labels_mfcc.txt")
    prediction_features = np.loadtxt("feature/wavenet_prediction_features_mfcc.txt")
    prediction_labels = np.loadtxt("feature/wavenet_prediction_labels_mfcc.txt")

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    loss_func = nn.CrossEntropyLoss()


    loss,accuracy,precision,recall,f1,auc,training_time = train_wavenet(model,rec_fields,train_features,train_labels,prediction_features,prediction_labels,
                                            optimizer,loss_func,args.batch_size,args.epochs,args.split_ratio)
    print ("training time " ,training_time)
    write_result(loss,accuracy,precision,recall,f1,auc,training_time,args.features_type,args.epochs,args.batch_size,args.split_ratio,args.arc)

elif args.features_type == 'raw_signal'and args.arc =='wavenet':
    ########### wavenet raw signal
    # wavenet_train_features_rawSignal,wavenet_train_labels_rawSignal = parse_audio_files_rawSignal(args.redimension_train_path,
                                                            # sub_dirs,file_ext,max_size,args.sample_rate)
    # wavenet_prediction_features_rawSignal,wavenet_prediction_labels_rawSignal = parse_audio_files_rawSignal(args.redimension_prediction_path,
                                                            # sub_dirs,file_ext,max_size,args.sample_rate)

    train_features = np.loadtxt("feature/wavenet_train_features_rawSignal.txt")
    train_labels = np.loadtxt("feature/wavenet_train_labels_rawSignal.txt")
    prediction_features = np.loadtxt("feature/wavenet_prediction_features_rawSignal.txt")
    prediction_labels = np.loadtxt("feature/wavenet_prediction_labels_rawSignal.txt")

    model,rec_fields,_ = wavenet()

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    loss,accuracy,precision,recall,f1,auc,training_time  = train_wavenet(model,rec_fields,train_features,train_labels,prediction_features,prediction_labels,
                                            optimizer,loss_func,args.batch_size,args.epochs,args.split_ratio)
    print ("training time " ,training_time)
    write_result(loss,accuracy,precision,recall,f1,auc,training_time,args.features_type,args.epochs,args.batch_size,args.split_ratio,args.arc)
