import argparse
import torch
import torch.optim as optim
import numpy as np
import os
import glob
import sh
import sys
import librosa

from read_csv import read,date_type
from util import clean,normaliser_features,encode_label
from redimension import read_audio, cut_padding_audio
from initial import main
from extration import parse_audio_files_mfcc,parse_audio_files_waveletPackets,parse_audio_files_rawSignal,get_cnn_mfccs




#####  Init: create the files
# main()

# cut files
parser = argparse.ArgumentParser(
    description='audio classification')

parser.add_argument('--train_wav_path', default='wavData/train',
                    help='path to the train wav data folder')

parser.add_argument('--train_csv_path', default='csvData/train',
                    help='path to the train csv data folder')

parser.add_argument('--read_path', default='read',
                    help='stock cut wav files')

parser.add_argument('--cut', default=False,
                    help='if you excute the code at first time. please cut the raw wav by input True')

parser.add_argument('--padding', default=False,
                    help='if you excute the code at first time. please padding the raw wav by input True')

parser.add_argument('--padding_path', default='padding',
                    help='path to stock padding audio')

parser.add_argument('--redimension_train_path', default='redimension/train',
                    help='path to redimension train wav folder')

parser.add_argument('--redimension_test_path', default='redimension/test',
                    help='path to redimension train wav folder')

parser.add_argument('--extract', default=False,
                    help='if you excute the code at first time. please extract features by input True')

parser.add_argument('--image_train_path', default='image/train',
                    help='path to stock train image folder')

parser.add_argument('--image_test_path', default='image/test',
                    help='path to stock train image folder')

args = parser.parse_args()
# print args.cut
# print args.train_csv_path
# print args.train_wav_path

########################   cut raw signal
if args.cut:
    clean(args.read_path)

    segProjet,segSite,segStart,duration,segLabel,segQuality = read(args.train_csv_path)
    # print segProjet
    date_type(args.train_wav_path,segProjet,segSite,segStart,duration,segLabel,segQuality,args.read_path)
    # print("ok")

#################  padding raw signal

sub_dirs = []
labels = []
for i, f in enumerate(glob.glob(args.read_path + os.sep +'*')):
    f = os.path.splitext(f)[0]
    sub_dirs.append(f.split(os.sep)[1])
    labels.append(f.split(os.sep)[1])
print labels

if args.padding:
    T_total = 4
    clean(args.redimension_train_path)
    clean(args.redimension_test_path)
    read_audio(args.read_path,sub_dirs,T_total,args.padding_path,labels)
    cut_padding_audio(args.padding_path,sub_dirs,T_total,labels,args.redimension_train_path,args.redimension_test_path)

if args.extract:
    ############ mfcc
    file_ext='*.wav'
    '''
    features_mfcc,labels_mfcc = parse_audio_files_mfcc(args.read_path,sub_dirs,file_ext)
    features_mfcc = normaliser_features(features_mfcc)

    labels_mfcc = encode_label(labels_mfcc)
    # print features_mfcc.shape
    # print labels_mfcc.shape
    np.savetxt("feature/train_features_mfcc.txt",features_mfcc)
    np.savetxt("feature/train_label_mfcc.txt",labels_mfcc)

    ############ wavelet
    features_wavelet,labels_wavelet = parse_audio_files_waveletPackets(args.read_path,sub_dirs,file_ext)
    features_wavelet = normaliser_features(features_wavelet)
    labels_wavelet = encode_label(labels_wavelet)

    np.savetxt("feature/train_features_wavelet.txt",features_wavelet)
    np.savetxt("feature/train_label_wavelet.txt",labels_wavelet)

    ###########  raw signal
    sample_size=200
    sample_rate=100
    features_rawSignal,labels_rawSignal = parse_audio_files_rawSignal(args.read_path,sub_dirs,file_ext,sample_size,sample_rate)
    features_rawSignal = normaliser_features(features_rawSignal)
    labels_rawSignal = encode_label(labels_rawSignal)

    np.savetxt("feature/train_features_rawSignal.txt",features_rawSignal)
    np.savetxt("feature/train_label_rawSignal.txt",labels_rawSignal)


    length = 28
    width = 28
    train_features,train_labels = get_cnn_mfccs(args.redimension_train_path,sub_dirs,file_ext,length,width)
    test_features,test_labels = get_cnn_mfccs(args.redimension_test_path,sub_dirs,file_ext,length,width)
    print train_features.shape,train_labels.shape
    # Write the array to disk
    with file('feature/cnn_train_features_mfcc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(train_features.shape))
        for data_slice in train_features:
            np.savetxt(outfile, data_slice, fmt='%-7.6f')
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    np.savetxt("feature/cnn_train_label_mfcc.txt",train_labels)

    # new_data = np.loadtxt('feature/cnn_train_features_mfcc.txt')
    # print new_data.shape
    # features = new_data.reshape((tra,5,10))
    '''

    rawSignal_to_image(args.redimension_train_path,sub_dirs,file_ext,args.image_train_path)
    rawSignal_to_image(args.redimension_test_path,sub_dirs,file_ext,args.image_test_path)

# parser.add_argument('--batch_size', type=int, default=100,
#                     metavar='N', help='training and valid batch size')
# parser.add_argument('--test_batch_size', type=int, default=100,
#                     metavar='N', help='batch size for testing')
# parser.add_argument('--arc', default='LeNet',
#                     help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19')
# parser.add_argument('--epochs', type=int, default=100,
#                     metavar='N', help='number of epochs to train')
# parser.add_argument('--lr', type=float, default=0.001,
#                     metavar='LR', help='learning rate')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     metavar='M', help='SGD momentum, for SGD only')
# parser.add_argument('--optimizer', default='adam',
#                     help='optimization method: sgd | adam')
# parser.add_argument('--cuda', default=True, help='enable CUDA')
# parser.add_argument('--seed', type=int, default=1234,
#                     metavar='S', help='random seed')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='num of batches to wait until logging train status')
# parser.add_argument('--patience', type=int, default=5, metavar='N',
# help='how many epochs of no loss improvement should we wait before stop training')
