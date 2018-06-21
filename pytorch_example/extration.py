import numpy as np
import sys
import os
import glob
import sh
import matplotlib.pyplot as plt
import pywt
import util
import librosa
import math
from sklearn import preprocessing


def extration_wavelet_packet(wavFile):

    sig, fs = librosa.load(wavFile)
    print("fs is ",fs)
    print("signal length is ", len(sig))
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

def extration_mfcc(wavFile):

    # Read the wav file
    X, sample_rate = librosa.load(wavFile)
    print("sample_rate  is ",sample_rate )
    print("data is ", X)

    # sftf
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T,axis=0)
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

def extration_rawSignal(wavFile,max_len,sample_rate):

    # Read the wav file
    X, sample_rate = librosa.load(wavFile, sr=sample_rate, mono=True)
    # print("sample_rate  is ",sample_rate )
    # print("data is ", X)
    # print X.shape
    if X.shape[0] < max_len:
        pad = np.zeros(max_len - X.shape[0])
        # print("pad is ",pad)
        X = np.hstack((X, pad))
    elif X.shape[0] > max_len:
        X = X[:max_len]
    # print X.shape
    return X


def parse_audio_files_rawSignal(redimension_dir,sub_read,file_ext):
    max_len=200
    sample_rate=100
    features, labels = np.empty((0,max_len)), np.empty(0)
    for label, sub_dir in enumerate(sub_read):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(redimension_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]
            try:
                audio = extration_rawSignal(f,max_len,sample_rate)

            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
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
    return np.array(features), np.array(labels, dtype = np.int)


def parse_audio_files_waveletPackets(read_dir,sub_read,file_ext):
    features, labels = np.empty((0,12)), np.empty(0)
    for label, sub_dir in enumerate(sub_read):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(read_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]
            quality = util.splitext(waveFile_name)[2]
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
    return np.array(features), np.array(labels, dtype = np.int)


def parse_audio_files_mfcc(read_dir,sub_read,file_ext):
    features, labels = np.empty((0,13)), np.empty(0)
    for label, sub_dir in enumerate(sub_read):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(read_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]
            quality = util.splitext(waveFile_name)[2]
            try:
                mfccs = extration_mfcc(f)
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
    return np.array(features), np.array(labels, dtype = np.int)


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

    read_dir = "read"
    sub_read = ['Bm','Eg']
    file_ext='*.wav'

    cmd = "rm -rf feature/*"
    sh.run(cmd)
    ############ mfcc
    features_mfcc,labels_mfcc = parse_audio_files_mfcc(read_dir,sub_read,file_ext)
    features_mfcc = normaliser_features(features_mfcc)
    # print ("features noramallisation are ",features_normalisation)
    labels_mfcc = encode_label(labels_mfcc)
    # print ("label encode is ",labels_encode)
    #print type(labels_encode)

    np.savetxt("feature/train_features_mfcc.txt",features_mfcc)
    np.savetxt("feature/train_label_mfcc.txt",labels_mfcc)

    ############ wavelet
    features_wavelet,labels_wavelet = parse_audio_files_waveletPackets(read_dir,sub_read,file_ext)
    features_wavelet = normaliser_features(features_wavelet)
    labels_wavelet = encode_label(labels_wavelet)

    np.savetxt("feature/train_features_wavelet.txt",features_wavelet)
    np.savetxt("feature/train_label_wavelet.txt",labels_wavelet)

    ###########  raw signal
    features_rawSignal,labels_rawSignal = parse_audio_files_rawSignal(read_dir,sub_read,file_ext)
    features_rawSignal = normaliser_features(features_rawSignal)
    labels_rawSignal = encode_label(labels_rawSignal)

    np.savetxt("feature/train_features_rawSignal.txt",features_rawSignal)
    np.savetxt("feature/train_label_rawSignal.txt",labels_rawSignal)
