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
from sklearn import metrics



def predict(features,labels,model_path,type):

    net2 = torch.load(model_path)
    prediction_x1, prediction_y1 = torch.from_numpy(features).float(), torch.from_numpy(labels).long()
    prediction_x = Variable(prediction_x1, requires_grad=False).type(torch.FloatTensor)
    prediction_y = prediction_y1.numpy().squeeze() # covert to numpy array
    if type =='mfcc':
        prediction_output = net2(prediction_x.view(-1,1,13))
    elif type =='wavelet':
        prediction_output = net2(prediction_x.view(-1,1,12))
    elif type == 'wavenet':
        prediction_output = net2(prediction_x.view(-1,1,200))

    pred_y = torch.max(prediction_output, 1)[1].data.numpy().squeeze()
    print("predicition is ",pred_y)
    print("origin data is ", prediction_y)
    accuracy = sum(pred_y == prediction_y) / float(prediction_y.size)
    print accuracy
    # print (metrics.classification_report(prediction_y, pred_y))


def clean(file_dir):
    for i, f in enumerate(glob.glob(file_dir + os.sep +'*')):
        # print f
        cmd = "rm -rf " + f  + "/*.wav"
        sh.run(cmd)

if __name__ == '__main__':
    #clean files

    # cmd = "rm -rf feature/*"
    # sh.run(cmd)

    ############### cut wav files

    sample_file_dir = "sample_csv/prediction"
    prediction_wav_dir = "prediction_wav"
    wav_dir = "wav/prediction"

    clean(prediction_wav_dir)

    segProjet,segSite,segStart,duration,segLabel,segQuality = read(sample_file_dir )
    date_type(wav_dir,segProjet,segSite,segStart,duration,segLabel,segQuality,prediction_wav_dir)

    ##############  extraction
    sub_dirs = ['Bm','Eg']
    file_ext='*.wav'
    ###############   mfcc
    prediction_features_mfcc,prediction_labels_mfcc = parse_audio_files_librosa(prediction_wav_dir,sub_dirs,file_ext)
    prediction_features_mfcc = normaliser_features(prediction_features_mfcc)

    prediction_labels_mfcc = encode_label(prediction_labels_mfcc)
    np.savetxt("feature/prediction_features_mfcc.txt",prediction_features_mfcc)
    np.savetxt("feature/prediction_labels_mfcc.txt",prediction_labels_mfcc)

    ###############   wavelet
    prediction_features_wavelet,prediction_labels_wavelet = parse_audio_files_waveletPackets(prediction_wav_dir,sub_dirs,file_ext)
    prediction_features_wavelet = normaliser_features(prediction_features_wavelet)

    prediction_labels_wavelet = encode_label(prediction_labels_wavelet)
    np.savetxt("feature/prediction_features_wavelet.txt",prediction_features_wavelet)
    np.savetxt("feature/prediction_labels_wavelet.txt",prediction_labels_wavelet)
    ###############   wavenet
    prediction_features_wavenet,prediction_labels_wavenet = parse_audio_files_wavenet(prediction_wav_dir,sub_dirs,file_ext)
    prediction_features_wavenet = normaliser_features(prediction_features_wavenet)

    prediction_labels_wavenet = encode_label(prediction_labels_wavenet)
    np.savetxt("feature/prediction_features_wavenet.txt",prediction_features_wavenet)
    np.savetxt("feature/prediction_labels_wavenet.txt",prediction_labels_wavenet)



    types = ['mfcc','wavelet','wavenet']
    features_mfcc = np.loadtxt("feature/prediction_features_mfcc.txt")
    labels_mfcc = np.loadtxt("feature/prediction_labels_mfcc.txt")
    # print np.unique(labels_mfcc)

    features_wavelet = np.loadtxt("feature/prediction_features_wavelet.txt")
    labels_wavelet = np.loadtxt("feature/prediction_labels_wavelet.txt")
    # print np.unique(labels_wavelet)

    features_wavenet = np.loadtxt("feature/prediction_features_wavenet.txt")
    labels_wavenet = np.loadtxt("feature/prediction_labels_wavenet.txt")
    # print np.unique(labels_wavenet)
    for type in types:
        print(type)
        if type =='mfcc':
            model_path = "model/mfcc_model_lstm.pkl"
            predict(features_mfcc,labels_mfcc,model_path,type)
        elif type == 'wavelet':
            model_path = "model/wavelet_model_lstm.pkl"
            predict(features_wavelet,labels_wavelet,model_path,type)
        elif type == 'wavenet':
            model_path = "model/wavenet_model_lstm.pkl"
            predict(features_wavenet,labels_wavenet,model_path,type)
