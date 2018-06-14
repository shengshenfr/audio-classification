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
import torch
from train_cnn import *

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
    totalNumOfFeatures = 28
    max_len = 28
    features = np.empty((0,totalNumOfFeatures,max_len))

    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    while (curPos + Win - 1 < N):
        countFrames += 1
        X = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position

        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)

        mfccs = librosa.feature.mfcc(y=X, sr=Fs, n_mfcc=totalNumOfFeatures)
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

        ext_features = np.hstack([mfccs])
        #print len(ext_features)
        # print ext_features.shape
        ext_features = np.resize(ext_features, (1, ext_features.shape[0],ext_features.shape[1]))
        features = np.vstack([features,ext_features])

    # print features
    print features.shape
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
    print ("length of signal",len(x))
    return x,Fs

def write_csv(wavFile,window_size,step_size,result):
    cmd = """ffmpeg -i """ + wavFile + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
    rs = sh.run(cmd, True)

    duration_in_wavFile = rs.stdout()
    name = wavFile.split(os.sep)[2]

    projet = name.split("_")[:3]
    date1 = name.split("_")[3]
    day = date1[-2:]
    date2 = name.split("_")[4].split(".")[0]
    second = date2[-2:]
    minite = date2[-4:-2]
    hour = date2[:2]


    len_result = len(result)

    str_start = '20'+date1[:2]+"/"+date1[-4:-2]+"/"+date1[-2:]+" "+date2[:2]+":"+date2[-4:-2]+":"+date2[-2:]
    start = pd.to_datetime(str_start,format='%Y/%m/%d %H:%M:%S')
    start_time = pd.date_range(start,periods = len_result,freq = '7s')
    print start_time[0]
    print window_size
    first_end_time = start_time[0]+1
    print first_end_time
    end_time = pd.date_range(first_end_time,periods = len_result,freq = '1min')

    csvFile = open("window/cnn_test.csv","w")
    writer = csv.writer(csvFile)
    fileHead = ["name","site","start_time","end_time","label"]
    writer.writerow(fileHead)
    for i in range(len_result):
        if result[i]==0:
            label = 'Ba'
        elif result[i]==1:
            label = 'Bm'
        else:
            label = 'Eg'
        d1 = [projet[0],projet[1],start_time[i],end_time[i],label]
        writer.writerow(d1)
    csvFile.close()



def predict(features,model):

    net2 = torch.load(model)
    x = torch.from_numpy(features).float()
    x = Variable(torch.unsqueeze(x, dim=1), requires_grad=False)
    test_output, _ = net2(x)
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    cmd = "rm -rf window/cnn_result.txt"
    sh.run(cmd)
    np.savetxt("window/cnn_result.txt",pred_y)

if __name__ == '__main__':

    wavFile = 'wav/prediction/WAT_OC_01_150520_000000.df100.x.wav'
    window_size = 7
    step_size  = 60
    '''
    x,Fs = read_audio(wavFile)
    signal= norm_signal(x)

    features = feature_extraction(signal,Fs,window_size,step_size)
    model_path = 'model/model_cnn.pkl'
    predict(features,model_path)
    '''
    result = np.loadtxt("window/cnn_result.txt")
    cmd = "rm -rf window/cnn_test.csv"
    sh.run(cmd)
    write_csv(wavFile,window_size,step_size,result)
