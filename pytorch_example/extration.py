import numpy as np
import sys
import os
import glob
import sh

import pywt
from util import clean_wav,clean_image
import librosa
import librosa.display

import matplotlib.pyplot as plt
import pylab

import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch

def extration_wavelet_packet(wavFile):

    sig, fs = librosa.load(wavFile)
    # print("fs is ",fs)
    # print("signal length is ", len(sig))
    N_signal = len(sig)
    n_level = 6
    #N_sub = N_signal/(2**n_level)
    #print sig
    #print ("N sub is ",N_sub)
    # x = sig.reshape((len(sig),1))
    # print x

    wp = pywt.WaveletPacket(data=sig, wavelet='db3', mode='symmetric')
    #
    # print wp.data
    #print wp.maxlevel
    #print len(wp.data)
    Node = []
    N_limit = 16
    limit = 0
    for node in wp.get_level(6, 'natural'):
            Node.append(node.path)

    print("sub sample is ",len(Node))

    TC = get_teager_energy(wp,Node,n_level,N_limit)
    print ("TC is ",TC)

    return TC


def  get_teager_energy(wp,Node,n,N_limit):
    k_len = 12
    TC = []
    for k in range(k_len):
        sum = 0
        for l in range(2**n):
            el = get_e(l,N_limit,wp,Node)
            sum += math.log(el,2) * math.cos(k*(l-0.5)*math.pi/(2**n))
        TC.append(sum)
    return TC

def get_e(l,N_limit,wp,Node):
    name = Node[l]
    x = wp[name].data
    sum = 0
    for t in range(N_limit):

        sum += x[t+1]*x[t+1]-x[t]*x[t+2]
        #print sum
    el = float(sum/N_limit)
    return el

def extration_mfcc(wavFile,mfcc_length):

    # Read the wav file
    X, sample_rate = librosa.load(wavFile)
    print("sample_rate  is ",sample_rate )
    print("data is ", X)

    # sftf
    # stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=mfcc_length).T,axis=0)

    '''
    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    return mfccs,chroma,mel,contrast
    '''
    return mfccs

def extration_rawSignal(wavFile,sample_size,sample_rate):

    # Read the wav file
    X, sample_rate = librosa.load(wavFile, sr=sample_rate, mono=True)
    # print("sample_rate  is ",sample_rate )
    # print("data is ", X)
    # print X.shape
    if X.shape[0] < sample_size:
        pad = np.zeros(sample_size - X.shape[0])
        # print("pad is ",pad)
        X = np.hstack((X, pad))
    elif X.shape[0] > sample_size:
        X = X[:sample_size]
    # print X.shape
    return X


def parse_audio_files_rawSignal(redimension_dir,sub_read,file_ext,sample_size,sample_rate):
    features, labels = np.empty((0,sample_size)), np.empty(0)
    for label, sub_dir in enumerate(sub_read):
        # print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(redimension_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]
            try:
                audio = extration_rawSignal(f,sample_size,sample_rate)

            except Exception as e:
                # print("[Error] extract feature error. %s" % (e))
                continue

            ext_features = np.hstack([audio])
            #print len(ext_features)
            #print ext_features
            features = np.vstack([features,ext_features])
            # labels = np.append(labels, quality)
            labels = np.append(labels, label)
        # print (features.shape)
        #print features
        #print labels

    features = normaliser_features(np.array(features))
    labels = encode_label(np.array(labels, dtype = np.int))

    return features,labels


def parse_audio_files_waveletPackets(read_dir,sub_read,file_ext):
    features, labels = np.empty((0,12)), np.empty(0)
    for label, sub_dir in enumerate(sub_read):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(read_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]

            try:
                TC = extration_wavelet_packet(f)

            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue

            ext_features = np.hstack([TC])
            #print len(ext_features)
            #print ext_features
            features = np.vstack([features,ext_features])
            # labels = np.append(labels, quality)
            labels = np.append(labels, label)
        print (features.shape)
        #print features
        #print labels
    features = normaliser_features(np.array(features))
    labels = encode_label(np.array(labels, dtype = np.int))
    return features,labels


def parse_audio_files_mfcc(read_dir,sub_read,file_ext,mfcc_length):
    features, labels = np.empty((0,mfcc_length)), np.empty(0)
    for label, sub_dir in enumerate(sub_read):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(read_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]

            try:
                mfccs = extration_mfcc(f,mfcc_length)
                # print len(mfccs),len(chroma),len(mel),len(contrast)
                # print (len(mfccs))
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
    features = normaliser_features(np.array(features))
    labels = encode_label(np.array(labels, dtype = np.int))
    return features, labels


def get_cnn_mfccs(redimension_dir,redimension_subs,file_ext,totalNumOfFeatures,max_len):

    features = np.empty((0,totalNumOfFeatures,max_len))

    labels = np.empty(0)
    for label, sub_dir in enumerate(redimension_subs):
        # print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(redimension_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]

            try:
                X, sample_rate = librosa.load(f)
                mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=totalNumOfFeatures)
                # print mfccs.shape
                # print ("mfcc is",mfccs)
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
            # print ext_features.shape
            ext_features = np.resize(ext_features, (1, ext_features.shape[0],ext_features.shape[1]))
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)
        # print (features.shape)
        # print (features)
        #print labels
    return np.array(features), np.array(labels, dtype = np.int)


def rawSignal_to_image(redimension_dir,redimension_subs,file_ext,image_dir):
    for label, sub_dir in enumerate(redimension_subs):
        # print("label: %s" % (label))
        # print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(redimension_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[3]
            spice = (os.path.splitext(f)[0]).split(os.sep)[2]
            # print waveFile_name
            sig, fs = librosa.load(f)


            pylab.axis('off') # no axis
            pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
            S = librosa.feature.melspectrogram(y=sig, sr=fs)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))



            # make pictures name
            save_path = image_dir+ "/" + spice + "/" + waveFile_name + '.png'
            pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
            pylab.close()

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




if __name__ == "__main__":

    redimension_train_path = "redimension/train"
    redimension_prediction_path = "redimension/prediction"
    sub_dirs = ['Ba','Bm','Eg']
    file_ext='*.wav'

    cmd = "rm -rf feature/*"
    sh.run(cmd)
    ############ lstm mfcc
    mfcc_length = 13
    lstm_train_features_mfcc,lstm_train_labels_mfcc = parse_audio_files_mfcc(redimension_train_path,sub_dirs,file_ext,mfcc_length)
    lstm_prediction_features_mfcc,lstm_prediction_labels_mfcc = parse_audio_files_mfcc(redimension_prediction_path,sub_dirs,file_ext,mfcc_length)

    np.savetxt("feature/lstm_train_features_mfcc.txt",lstm_train_features_mfcc)
    np.savetxt("feature/lstm_train_label_mfcc.txt",lstm_train_labels_mfcc)
    np.savetxt("feature/lstm_prediction_features_mfcc.txt",lstm_prediction_features_mfcc)
    np.savetxt("feature/lstm_prediction_labels_mfcc.txt",lstm_prediction_labels_mfcc)

    ############ lstm wavelet
    lstm_train_features_wavelet,lstm_train_labels_wavelet = parse_audio_files_waveletPackets(redimension_train_path,sub_dirs,file_ext)
    lstm_prediction_features_wavelet,lstm_prediction_labels_wavelet = parse_audio_files_waveletPackets(redimension_prediction_path,sub_dirs,file_ext)

    np.savetxt("feature/lstm_train_features_wavelet.txt",lstm_train_features_wavelet)
    np.savetxt("feature/lstm_train_labels_wavelet.txt",lstm_train_labels_wavelet)
    np.savetxt("feature/lstm_prediction_features_wavelet.txt",lstm_prediction_features_wavelet)
    np.savetxt("feature/lstm_prediction_labels_wavelet.txt",lstm_prediction_labels_wavelet)

    ########### lstm raw signal
    sample_size=200
    sample_rate=100
    lstm_train_features_rawSignal,lstm_train_labels_rawSignal = parse_audio_files_rawSignal(redimension_train_path,sub_dirs,file_ext,sample_size,sample_rate)
    lstm_prediction_features_rawSignal,lstm_prediction_labels_rawSignal = parse_audio_files_rawSignal(redimension_train_path,sub_dirs,file_ext,sample_size,sample_rate)

    np.savetxt("feature/lstm_train_features_rawSignal.txt",lstm_train_features_rawSignal)
    np.savetxt("feature/lstm_train_labels_rawSignal.txt",lstm_train_labels_rawSignal)
    np.savetxt("feature/lstm_prediction_features_rawSignal.txt",lstm_prediction_features_rawSignal)
    np.savetxt("feature/lstm_prediction_labels_rawSignal.txt",lstm_prediction_labels_rawSignal)

    ################   cnn mfcc

    length = 28
    width = 28
    cnn_train_features_mfcc,cnn_train_labels_mfcc = get_cnn_mfccs(redimension_train_path,sub_dirs,file_ext,length,width)
    cnn_prediction_features_mfcc,cnn_prediction_labels_mfcc = get_cnn_mfccs(redimension_prediction_path,sub_dirs,file_ext,length,width)
    # Write the array to disk
    with file('feature/cnn_train_features_mfcc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(cnn_train_features_mfcc.shape))
        for data_slice in cnn_train_features_mfcc:
            np.savetxt(outfile, data_slice, fmt='%-7.6f')
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    np.savetxt("feature/cnn_train_label_mfcc.txt",cnn_train_labels_mfcc)

    with file('feature/cnn_prediction_features_mfcc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(cnn_prediction_features_mfcc.shape))
        for data_slice in cnn_prediction_features_mfcc:
            np.savetxt(outfile, data_slice, fmt='%-7.6f')
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    np.savetxt("feature/cnn_prediction_labels_mfcc.txt",cnn_prediction_labels_mfcc)



    ######  cnn raw signal
    image_train_path = "image/train"
    image_prediction_path = "image/prediction"

    clean_image(image_train_path)
    clean_image(image_prediction_path)
    rawSignal_to_image(redimension_train_path,sub_dirs,file_ext,image_train_path)
    rawSignal_to_image(redimension_prediction_path,sub_dirs,file_ext,image_prediction_path)
