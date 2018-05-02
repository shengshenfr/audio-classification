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


def format_array(arr):
    return "[%s]" % ", ".join(["%.14f" % x for x in arr])

def extration_wavelet_packet(wavFile):
    wavelet = pywt.Wavelet('db1')
    print(format_array(wavelet.dec_lo), format_array(wavelet.dec_hi))
    wavelet = pywt.Wavelet('db2')
    print(format_array(wavelet.dec_lo), format_array(wavelet.dec_hi))
    wavelet = pywt.Wavelet('db3')
    print(format_array(wavelet.dec_lo), format_array(wavelet.dec_hi))

    '''
    sig, fs = librosa.load(wavFile)
    print("fs is ",fs)
    print("signal length is ", len(sig))
    N_signal = len(sig)
    n_level = 3
    N_sub = N_signal/(2**n_level)
    print sig
    # x = sig.reshape((len(sig),1))
    # print x
    wp = pywt.WaveletPacket(data=sig, wavelet='db3', mode='symmetric')

    print wp.data
    print wp.maxlevel
    # print len(wp.data)
    '''
def extration_wavelet(wavFile):


    in_data, fs = librosa.load(wavFile)
    #print("fs is ",fs)
    print("data is ", in_data)

    coef, freqs = pywt.cwt(in_data,np.arange(1,129),'gaus1')
    #print coef
    print len(coef)
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




def parse_audio_files(result_redimension_dir,sub_redimensions,file_ext):
    features = np.array([])
    for i, sub_dir in enumerate(sub_redimensions):
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(result_redimension_dir, sub_dir, file_ext)):
            print("extract file: %s" % (f))
            try:
                #mfccs,chroma,mel,contrast = extration_librosa(f)
                extration_wavelet_packet(f)
                #print ("mfcc is",np.array(mfccs))
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue

            # ext_features = np.concatenate((np.array(mfccs),np.array(chroma)),axis=0)
            # print len(ext_features)
            #print ext_features
            #features = np.vstack([features,ext_features])
            #labels = np.append(labels, f.split('/')[-1].split('-')[1])

    #return np.array(features)

if __name__ == "__main__":

    result_redimension_dir = "result_redimension"
    sub_redimensions = ['bm_redimension', 'eg_redimension']
    file_ext='*.wav'
    parse_audio_files(result_redimension_dir,sub_redimensions,file_ext)
