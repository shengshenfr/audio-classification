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
from sklearn import preprocessing
import torch
import util
from torch.autograd import Variable

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
        # countFrames += 1
        X = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position

        # if countFrames == 1:
        #     Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)

        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=Fs, n_mfcc=13),axis=1)
        ext_features = np.hstack([mfccs])
        #print len(ext_features)
        #print ext_features
        features = np.vstack([features,ext_features])


        # print mfccs
        # print curFV
    print features.shape
    # print features
    return features


def normaliser_features(features):

    features_normalisation = preprocessing.scale(features)

    return features_normalisation

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
    # print("the duration of wavfile is ", duration_in_wavFile)
    # print wavFile
    name = wavFile.split(os.sep)[2]
    # print name

    projet = name.split("_")[:3]
    # print projet
    date1 = name.split("_")[3]
    day = date1[-2:]
    date2 = name.split("_")[4].split(".")[0]
    second = date2[-2:]
    minite = date2[-4:-2]
    hour = date2[:2]
    # print date1
    # print day
    # print date2,hour,minite,second


    len_result = len(result)
    # print int(duration_in_wavFile)/3600
    str_start = '20'+date1[:2]+"/"+date1[-4:-2]+"/"+date1[-2:]+" "+date2[:2]+":"+date2[-4:-2]+":"+date2[-2:]
    # print str_start
    start = pd.to_datetime(str_start,format='%Y/%m/%d %H:%M:%S')
    # print start
    start_time = pd.date_range(start,periods = len_result,freq = '1s')
    # print start_time
    # print type(start_time)
    # print len(start_time)
    first_end_time = start_time[0]+1
    # print first_end_time
    end_time = pd.date_range(first_end_time,periods = len_result,freq = '1min')
    # print end_time
    csvFile = open("window/lstm_test.csv","w")
    writer = csv.writer(csvFile)
    fileHead = ["name","site","start_time","end_time","label"]
    writer.writerow(fileHead)
    for i in range(len_result):
        if result[i]==0:
            label = 'Bm'
        elif result[i]==1:
            label = 'Eg'

        d1 = [projet[0],projet[1],start_time[i],end_time[i],label]
        writer.writerow(d1)
    csvFile.close()
        # dataframe = pd.DataFrame({'name':projet[0],'site':projet[1],'start_time':start_time[i],'end_time':end_time[i],'label':label},index=[i])
        # dataframe.to_csv("window/test.csv",index = False,sep=',')

def predict(features_normalisation,model):

    net2 = torch.load(model)
    prediction_x1 = torch.from_numpy(features_normalisation).float()
    prediction_x = Variable(prediction_x1, requires_grad=False).type(torch.FloatTensor)

    prediction_output = net2(prediction_x.view(-1,1,13))
    # prediction_output = net2(prediction_x.view(-1,1,12))
    # 0 is Ba, 1 is Bm, 2 is Eg
    pred_y = torch.max(prediction_output, 1)[1].data.numpy().squeeze()
    print("predicition is ",pred_y)
    cmd = "rm -rf window/lstm_result.txt"
    sh.run(cmd)
    np.savetxt("window/lstm_result.txt",pred_y)
    # print("origin data is ", prediction_y)
    # accuracy = sum(pred_y == prediction_y) / float(prediction_y.size)
    # print accuracy




if __name__ == '__main__':

    wavFile = 'wavData/prediction/HAT_A_02_121023_142845.d100.x.wav'
    window_size = 1
    step_size  = 60

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
    features = feature_extraction(signal,Fs,window_size,step_size)
    features_normalisation = normaliser_features(features)
    # cmd = "rm -rf feature/window.txt"
    # sh.run(cmd)
    np.savetxt("feature/window.txt",features_normalisation)

    features_normalisation = np.loadtxt("feature/window.txt")
    model_path = 'model/mfcc_model_lstm.pkl'
    # model_path = 'model/rnn_wavelet.pkl'
    predict(features_normalisation,model_path)
    cmd = "rm -rf window/lstm_test.csv"
    sh.run(cmd)
    result = np.loadtxt("window/lstm_result.txt")
    write_csv(wavFile,window_size,step_size,result)
    '''
    wav_prediction_dir = 'wav/prediction'
    file_ext = '*.wav'
    for wavFile in glob.glob(os.path.join(wav_prediction_dir, file_ext)):
        x,Fs = read_audio(wavFile)
        feature_extraction(x,Fs,window_size,step_size)
    '''
