from python_speech_features import mfcc
from python_speech_features import logfbank

import scipy.io.wavfile as wav
import numpy as np
import sh
import matplotlib.pyplot as plt
import os



def get_sample_feature(input_mfcc_feature):
    mfcc_feature = np.loadtxt(input_mfcc_feature)
    sample_size = len(mfcc_feature)
    time_step_s = 100
    sample_per_s = sample_size / time_step_s

    #time_window = 1
    sample_data = []
    print(len(mfcc_feature[0]))
    for j in range(0,len(mfcc_feature[0])):
        #print(j)
        temp_data = []
        for i in range(0,time_step_s,time_window):

            #temp_data.append(np.std(mfcc_feature[i*sample_per_s:(i+time_window)*sample_per_s,j]))


            sample_data.append(temp_data)
    print(np.shape(sample_data))

    return np.transpose(sample_data)



def read_sample_feature(mfcc_dir,sample_feature_file):
    mfcc_feature_list = os.listdir(mfcc_dir)
    print(mfcc_feature_list)
    sample_data = []

    for file_name in mfcc_feature_list:
        print(mfcc_dir + "/" + file_name)
        temp = get_sample_feature(mfcc_dir + "/" + file_name)
        if len(sample_data) == 0:
            sample_data = temp

        else:
            sample_data = np.concatenate((sample_data,temp))


    print(np.shape(sample_data))
    np.savetxt(sample_feature_file,sample_data)



def plot_sample(sample_data,subplot_cnt):
    mp3_file = "mp3/sound.mp3"
    cmd = """ffmpeg -i """ + mp3_file + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
    rs = sh.run(cmd, True)
    duration_in_s = rs.stdout()
    print(duration_in_s)
    print("total duration is " + duration_in_s + "s")


    t = np.linspace(0, int(float(duration_in_s)), len(sample_data))
    #print("t is " , t)
    print("t is " , len(t))
    plt.figure()
    for i in range(1,subplot_cnt+1):
        plt.subplot(subplot_cnt,1,i)
        plt.plot(t,sample_data[:,i])
    plt.show()


if __name__ == '__main__':
    input_mfcc_feature = "feature/0.txt"

    #get_sample_feature(input_mfcc_feature)


    sample_txt = "sample/sample.txt"
    sample_data = np.loadtxt(sample_txt)
    plot_sample(sample_data,3)
