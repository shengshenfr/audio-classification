import sys
import numpy as np
import os
import csv
import glob
import pandas as pd
import sh

import scipy.io.wavfile as wavfile
import pydub
from pydub import AudioSegment
import librosa



def norm_signal(signal):

    # Signal normalization
    signal = np.double(signal)
    # print signal
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)
    print signal
    return signal

def feature_extraction(signal,Fs,window_size,step_size):
    Win = int(round(window_size*Fs))
    Step = int(round(step_size*Fs))
    print("window,step  ",Win,Step)
    features = np.empty((0,13))
    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0

    totalNumOfFeatures = 13

    while (curPos + Win - 1 < N):
        countFrames += 1
        X = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position

        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)

        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=Fs, n_mfcc=13),axis=1)
        ext_features = np.hstack([mfccs])
        #print len(ext_features)
        #print ext_features
        features = np.vstack([features,ext_features])


        # print mfccs
        # print curFV
    print features.shape
    print features
    return features



def read_audio(wavFile):

    audiofile = AudioSegment.from_file(wavFile)
    # print audiofile.sample_width
    if audiofile.sample_width==2:
        data = np.fromstring(audiofile._data, np.int16)
    elif audiofile.sample_width==4:
        data = np.fromstring(audiofile._data, np.int32)

    Fs = audiofile.frame_rate
    x = []
    for chn in xrange(audiofile.channels):
        x.append(data[chn::audiofile.channels])
    x = np.array(x).T
    # print x.shape
    # print x
    if x.ndim==2:
        if x.shape[1]==1:
            x = x.flatten()
    # print x
    print len(x)
    return x,Fs
'''
def cut_audio(wavFile):
    cmd = """ffmpeg -i """ + wavFile + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
    rs = sh.run(cmd, True)

    duration_in_wavFile = rs.stdout()
    print("the duration of wavfile is ", int(duration_in_wavFile)/3600)
    # print wavFile
    # name = wavFile.split(os.sep)[2]
    # print name

    projet = name.split("_")[:3]
    print projet
    date1 = name.split("_")[3]
    day = date1[-2:]
    date2 = name.split("_")[4].split(".")[0]
    second = date2[-2:]
    minite = date2[-4:-2]
    hour = date2[:2]
    print date1
    print day
    print date2,len(hour),minite,second

    cut_point = np.linspace(0,int(duration_in_wavFile)/3600,int(duration_in_wavFile)/3600+1)
    print cut_point
    dur = 3600
    path_name = "window/cut/"
    for i in range(int(duration_in_wavFile)/3600):
        if i/24==0:
            if len(str(i))==1:
                hour= "0"+str(i)
            else:
                hour = str(i)
            name = "WAT_BP_01_161002_"+hour+"0000"
        else i/24==1:
            if i ==24:
                name = "WAT_BP_01_161003_000000"
            else:
                name = "WAT_BP_01_161003_"+hour+"000000"
        # cmd = "ffmpeg -ss " + str(cut_point[i]) + " -t " + str(dur)+ " -i " + wavFile + " " + path_name + waveFile_name +"_"+ str(j) + ".wav"
        # sh.run(cmd)
'''


if __name__ == '__main__':

    wavFile = 'wav/prediction/WAT_OC_01_150520_000000.df100.x.wav'
    window_size = 1
    step_size  = 60
    # cut_audio(wavFile)
    x,Fs = read_audio(wavFile)
    signal= norm_signal(x)
    # signal_txt = "feature/signal.txt"
    # Fs_txt = "feature/Fs.txt"
    #
    # np.savetxt(signal_txt, signal)
    # np.savetxt(Fs_txt, Fs)
    #
    # signal = np.loadtxt(signal_txt)
    # Fs = np.loadtxt(Fs_txt)
    feature_extraction(signal,Fs,window_size,step_size)






    '''
    wav_prediction_dir = 'wav/prediction'
    file_ext = '*.wav'
    for wavFile in glob.glob(os.path.join(wav_prediction_dir, file_ext)):
        x,Fs = read_audio(wavFile)
        feature_extraction(x,Fs,window_size,step_size)
    '''
