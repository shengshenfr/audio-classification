import numpy as np
from sys import argv
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


def format_array(arr):
    return "[%s]" % ", ".join(["%.14f" % x for x in arr])

def extration_wavelet_packet(wavFile):
    # wavelet = pywt.Wavelet('db1')
    # print(format_array(wavelet.dec_lo), format_array(wavelet.dec_hi))
    # wavelet = pywt.Wavelet('db2')
    # print(format_array(wavelet.dec_lo), format_array(wavelet.dec_hi))
    # wavelet = pywt.Wavelet('db3')
    # print(format_array(wavelet.dec_lo), format_array(wavelet.dec_hi))


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

def extration_wavelet(wavFile):


    in_data, fs = librosa.load(wavFile)
    #print("fs is ",fs)
    print("data is ", in_data)

    coef, freqs = pywt.cwt(in_data,np.arange(1,129),'gaus1')
    #print coef
    #print len(coef)
    # plt.imshow(coef, extent=[-1, 1, 1, 128], cmap='PRGn', aspect='auto',
    #       vmax=abs(coef).max(), vmin=-abs(coef).max())
    # plt.show()

def extration_librosa(wavFile):

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


def parse_audio_files_librosa(read_dir,sub_read,file_ext):
    features, labels = np.empty((0,13)), np.empty(0)
    for label, sub_dir in enumerate(sub_read):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(read_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]
            quality = util.splitext(waveFile_name)[2]
            try:
                mfccs = extration_librosa(f)
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
    '''
    redimension_dir = "redimension"
    sub_redimensions = ['Ba', 'Bm','Eg']
    file_ext='*.wav'
    parse_audio_files_librosa(redimension_dir,sub_redimensions,file_ext)
    parse_audio_files_waveletPackets(redimension_dir,sub_redimensions,file_ext)
    '''
    read_dir = "read"
    sub_read = ['Ba', 'Bm','Eg']
    file_ext='*.wav'
    features,labels = parse_audio_files_librosa(read_dir,sub_read,file_ext)
    features_normalisation = normaliser_features(features)
    # print ("features noramallisation are ",features_normalisation)
    labels_encode = encode_label(labels)
    print ("label encode is ",labels_encode)
    #print type(labels_encode)
    cmd = "rm -rf feature/*"
    sh.run(cmd)
    np.savetxt("feature/prediction_features.txt",features_normalisation)
    np.savetxt("feature/prediction_label.txt",labels_encode)

    # parse_audio_files_waveletPackets(read_dir,sub_read,file_ext)
