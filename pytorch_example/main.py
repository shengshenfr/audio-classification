
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
from util import cut,extract,train_model

from initial import main


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
parser.add_argument('--extract', default=False,
                    help='if you excute the code at first time. please extract features by input True')

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
                    help='network architecture: lstm,cnn,wavenet,svm,knn,hmm,alexnet')

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

if args.extract:

    extract(args.read_path,args.redimension_train_path,args.redimension_validation_path,args.mfcc_length,
                        args.sample_size,args.sample_rate,args.length,args.width,args.image_train_path,args.image_validation_path)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~Loading data

# # types = ['mfcc','wavelet','rawSignal']
train_model(args.features_type,args.arc,args.hidden_size,args.num_layers,args.num_classes,
                args.drop_out,args.lr,args.batch_size,args.epochs,args.split_ratio,args.length,
                    args.width,args.momentum,args.optimizer,args.image_train_path,args.image_validation_path)
