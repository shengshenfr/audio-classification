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

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    return mfccs,chroma,mel,contrast




def parse_audio_files_librosa(result_redimension_dir,sub_redimensions,file_ext):
    features, labels = np.empty((0,160)), np.empty(0)
    for label, sub_dir in enumerate(sub_redimensions):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(result_redimension_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            waveFile_name = (os.path.splitext(f)[0]).split(os.sep)[2]
            quality = util.splitext(waveFile_name)[2]
            try:
                mfccs,chroma,mel,contrast = extration_librosa(f)
                # extration_wavelet_packet(f)
                #print len(mfccs),len(chroma),len(mel),len(contrast)
                #print ("mfcc is",np.array(mfccs))
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue

            ext_features = np.hstack([mfccs,chroma,mel,contrast])
            #print len(ext_features)
            #print ext_features
            features = np.vstack([features,ext_features])
            # labels = np.append(labels, quality)
            labels = np.append(labels, label)
        #print features.shape
        #print labels
    return np.array(features), np.array(labels, dtype = np.int)

if __name__ == "__main__":

    result_redimension_dir = "result_redimension"
    sub_redimensions = ['bm_redimension', 'eg_redimension']
    file_ext='*.wav'
    parse_audio_files_librosa(result_redimension_dir,sub_redimensions,file_ext)
