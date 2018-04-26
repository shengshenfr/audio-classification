import numpy as np
from sys import argv
import sys
import os
import glob
import sh
import matplotlib.pyplot as plt
import pywt
import librosa


def audio(wav_dir):
    for i, f in enumerate(glob.glob(wav_dir + os.sep +'*.wav')):               # for each WAV file
        wavFile = f
        #print wavFile
        # Read the wav file
        in_data, fs = librosa.load(wavFile)
        #print("fs is ",fs)
        print("data is ", in_data)

        coeffs = pywt.wavedec(in_data,"db1",level=3)
        print len(coeffs[0])







if __name__ == "__main__":

    result_padding_dir = "test_wav"
    audio(result_padding_dir)
