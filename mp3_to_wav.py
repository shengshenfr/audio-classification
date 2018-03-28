from python_speech_features import mfcc
from python_speech_features import logfbank

import scipy.io.wavfile as wav
import numpy as np
import sh
import matplotlib.pyplot as plt
import os


def mfcc_feature(input_wav, output_txt):
    (rate,sig) = wav.read(input_wav)
    mfcc_feat = mfcc(sig,rate)

    np.savetxt(output_txt, mfcc_feat)
    print(len(mfcc_feat))

    fbank_feat = logfbank(sig,rate)
    print(fbank_feat[1:3,:])


def run(mp3_file, wav_dir, mfcc_feature_dir):
    print(mp3_file)
    cmd = """ffmpeg -i """ + mp3_file + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
    rs = sh.run(cmd, True)
    duration_in_s = rs.stdout()
    print(duration_in_s)
    print("total duration is " + duration_in_s + "s")
    step_s = 600

    for i in range(0, int(float(duration_in_s)), step_s):
        wav_file = wav_dir + "/" + str(i) + ".wav"
        mfcc_feature_file = mfcc_feature_dir + "/" + str(i) + ".txt"
        cmd = "ffmpeg -ss " + str(i) + " -t " + str(step_s) + " -i " + mp3_file + " -f wav " + wav_file
        sh.run(cmd)
        print("convert mp3 to wav finish, " + str(i))
        # extract feature
        mfcc_feature(wav_file, mfcc_feature_file)

        # clean files
        cmd = "rm -rf " + wav_dir  + "/*"
        sh.run(cmd)


def plot(data,subplot_cnt):
    mp3_file = "mp3/music.mp3"
    cmd = """ffmpeg -i """ + mp3_file + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
    rs = sh.run(cmd, True)
    duration_in_s = rs.stdout()
    print(duration_in_s)
    print("total duration is " + duration_in_s + "s")


    t = np.linspace(0, int(float(duration_in_s)), len(data))
    print("t is " , t)
    print("t is " , len(t))
    plt.figure()
    for i in range(1,subplot_cnt+1):
        plt.subplot(subplot_cnt,1,i)
        plt.plot(t,data[:,i])
    plt.show()

if __name__ == '__main__':
    feature_txt = "feature/0.txt"
    feature_data = np.loadtxt(feature_txt)
    plot(feature_data,3)
