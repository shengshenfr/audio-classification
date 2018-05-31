import sys
import numpy as np
import os
import csv
import glob
import pandas as pd
import sh
import librosa
import wave
import scipy.io.wavfile as wavfile
import pydub
from pydub import AudioSegment


def feature_extraction(signal,sample_rate,window_size,step_size):
    win = int(window_size)
    step = int(step_size)


    # Signal normalization
    signal = np.double(signal)
    print signal
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)
    print signal

if __name__ == '__main__':
    wav_prediction_dir = 'wav/prediction'
    # wav_prediction_dir  = 'read/Ba'
    window_size = 30
    step_size = 1
    file_ext = '*.wav'
    for wavFile in glob.glob(os.path.join(wav_prediction_dir, file_ext)):
        # X, sample_rate = librosa.load(wavFile)
        # print X

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

        feature_extraction(x,Fs,window_size,step_size)
