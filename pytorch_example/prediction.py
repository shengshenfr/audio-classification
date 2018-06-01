import sys
import os
import csv
import glob
import pandas as pd
import sh

import numpy as np
from read_csv import *
from train_lstm import *
import torch



def date_type(wav_dir,segProjet,segSite,segStart,duration,segLabel,segQuality,prediction_wav_dir):

    for i, f in enumerate(glob.glob(wav_dir + os.sep +'*.wav')):               # for each WAV file
        wavFile = f
        print os.path.splitext(wavFile)[0]
        waveFile_name = (os.path.splitext(wavFile)[0]).split(os.sep)[3]
        print waveFile_name
        date1 = waveFile_name.split("_")[3]
        date2 = (waveFile_name.split("_")[4]).split(".")[0]
        #print date1,date2
        temp1 = []
        temp2 = []
        for i in range(len(date1)):
            #print date1[i], '(%d)' %i
            temp1.append(date1[i])
        date1 = "20" + temp1[0] + temp1[1] +"-"+ temp1[2] + temp1[3] +"-"+ temp1[4]+temp1[5]
        print ("date1 is ",date1)

        for j in range(len(date2)):
            #print date2[j], '(%d)' %j
            temp2.append(date2[j])
        date2 = temp2[0] + temp2[1] +":"+ temp2[2] + temp2[3] +":"+ temp2[4]+temp2[5]
        #print date2

        date_str = date1 +" "+ date2
        #print data_str

        start_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

        cut(wavFile, segProjet,segSite,segStart,duration,segLabel,segQuality,prediction_wav_dir,start_date)



def prepration():

    sample_file_dir = "sample_csv/WAT"
    prediction_wav_dir = "prediction_wav"
    wav_dir = "wav/prediction"
    # result_eg_dir = "prediction_wav/eg"

    #clean files
    # cmd = "rm -rf " + result_bm_dir  + "/*"
    # sh.run(cmd)
    # cmd = "rm -rf " + result_eg_dir  + "/*"
    # sh.run(cmd)
    # cmd = "rm -rf " + combine_dir  + "/*"
    # sh.run(cmd)
    for i, f in enumerate(glob.glob(sample_file_dir + os.sep +'*.csv')):
        print f
        f_name = (os.path.splitext(f)[0]).split(os.sep)[2]
        print f_name
        f_site = f_name.split("_")[1]
        print f_site
        for j,w in enumerate (glob.glob(wav_dir + os.sep + '*' )):
            print ("w is ",w)
            w_name = (os.path.splitext(w)[0]).split(os.sep)[2]
            print w_name
            w_site = w_name.split("_")[1]

            if f_site == w_site:
                print ("ok")
                segProjet,segSite,segStart,duration,segLabel,segQuality = read(f)
                date_type(w,segProjet,segSite,segStart,duration,segLabel,segQuality,prediction_wav_dir)

def get_features():

    prediction_wav_dir = "prediction_wav"
    sub_dirs = ['Bm','Eg']
    file_ext='*.wav'
    features,labels = parse_audio_files_librosa(prediction_wav_dir,sub_dirs,file_ext)
    # features,labels = parse_audio_files_waveletPackets(prediction_dir,sub_dir,file_ext)
    features_normalisation = normaliser_features(features)
    # print ("features noramallisation are ",features_normalisation)

    labels_encode = encode_label(labels)


    np.savetxt("feature/prediction_features.txt",features_normalisation)
    np.savetxt("feature/prediction_label.txt",labels_encode)

def predict():
    features_normalisation = np.loadtxt("feature/prediction_features.txt")
    labels_encode = np.loadtxt("feature/prediction_label.txt")

    model_path = 'model/rnn.pkl'
    # model_path = 'model/rnn_wavelet.pkl'
    net2 = torch.load(model_path)
    prediction_x1, prediction_y1 = torch.from_numpy(features_normalisation).float(), torch.from_numpy(labels_encode).long()
    prediction_x = Variable(prediction_x1, requires_grad=False).type(torch.FloatTensor)
    prediction_y = prediction_y1.numpy().squeeze() # covert to numpy array

    prediction_output = net2(prediction_x.view(-1,1,13))
    # prediction_output = net2(prediction_x.view(-1,1,12))

    pred_y = torch.max(prediction_output, 1)[1].data.numpy().squeeze()
    print("predicition is ",pred_y)
    print("origin data is ", prediction_y)
    accuracy = sum(pred_y == prediction_y) / float(prediction_y.size)
    print accuracy



if __name__ == '__main__':
    #clean files
    '''
    cmd = "rm -rf feature/*"
    sh.run(cmd)
    prepration()
    get_features()
    '''
    predict()
