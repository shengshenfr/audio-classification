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

    time_window = 1
    sample_data = []
    print(len(mfcc_feature[0]))
    for j in range(0,len(mfcc_feature[0])):
        #print(j)
        temp_data = []
        for i in range(0,time_step_s,time_window):

            temp_data.append(np.std(mfcc_feature[i*sample_per_s:(i+time_window)*sample_per_s,j]))


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






if __name__ == '__main__':
    input_mfcc_feature = "feature/0.txt"

    sample_feature(input_mfcc_feature)
