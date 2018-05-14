import sys
import os
import csv
import glob
import pandas as pd
import sh

import numpy as np
from read_csv import *
from train import *
import torch

if __name__ == '__main__':

    sample_file = "sample/HAT_A_LF_dev.csv"

    wav_dir = "wav/prediction"

    result_bm_dir = "prediction_wav/bm"
    result_eg_dir = "prediction_wav/eg"

    #clean files
    cmd = "rm -rf " + result_bm_dir  + "/*"
    sh.run(cmd)
    cmd = "rm -rf " + result_eg_dir  + "/*"
    sh.run(cmd)
    # cmd = "rm -rf " + combine_dir  + "/*"
    # sh.run(cmd)

    segProjet,segSite,segStart,duration,segLabel,segQuality = read(sample_file)
    date_type(wav_dir,segProjet,segSite,segStart,duration,segLabel,segQuality,result_bm_dir,result_eg_dir)


    prediction_dir = "prediction_wav"
    sub_read = ['bm', 'eg']
    file_ext='*.wav'
    features,labels = parse_audio_files_librosa(prediction_dir,sub_read,file_ext)

    features_normalisation = normaliser_features(features)
    # print ("features noramallisation are ",features_normalisation)

    labels_encode = encode_label(labels)

    model_path = 'model/rnn.pkl'
    net2 = torch.load(model_path)
    prediction_x1, prediction_y1 = torch.from_numpy(features_normalisation).float(), torch.from_numpy(labels_encode).long()
    prediction_x = Variable(prediction_x1, requires_grad=True).type(torch.FloatTensor)
    prediction_y = prediction_y1.numpy().squeeze() # covert to numpy array

    prediction_output = net2(prediction_x.view(-1,1,13))
    pred_y = torch.max(prediction_output, 1)[1].data.numpy().squeeze()
    print("predicition is ",pred_y)
    print("origin data is ", prediction_y)
    accuracy = sum(pred_y == prediction_y) / float(prediction_y.size)
    print accuracy
