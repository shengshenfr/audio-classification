import sys
import os
import glob
import csv
import os.path
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa.display
from scipy import signal
import sh
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,f1_score,recall_score

from extration import parse_audio_files_mfcc,parse_audio_files_waveletPackets,parse_audio_files_rawSignal,get_cnn_mfccs,rawSignal_to_image
from redimension import read_audio,distribuer
from read_csv import read,date_type
from test_wavenet import wavenet,train_wavenet

from train_lstm import RNN,train_rnn
from train_cnn import CNN,train_mfcc,train_rawSignal,predict_rawSignal

import torch
import torch.optim as optim
from torch import nn

def get_spices(read_path):
    sub_dirs = []
    labels = []
    for i, f in enumerate(glob.glob(read_path + os.sep +'*')):
        f = os.path.splitext(f)[0]
        sub_dirs.append(f.split(os.sep)[1])
        labels.append(f.split(os.sep)[1])
    # print labels
    return sub_dirs,labels


def cut(read_path,train_csv_path,train_wav_path,redimension_train_path,redimension_validation_path,padding_path,split_ratio):
    clean_wav(read_path)
    clean_wav(padding_path)
    clean_wav(redimension_train_path)
    clean_wav(redimension_validation_path)

    segProjet,segSite,segStart,duration,segLabel,segQuality = read(train_csv_path)
    # print segProjet
    max_duration = date_type(train_wav_path,segProjet,segSite,segStart,duration,segLabel,segQuality,read_path)
    # print("ok")

    #################  padding raw signal
    file_ext='*.wav'
    sub_dirs,labels = get_spices(read_path)

    read_audio(read_path,sub_dirs,max_duration,padding_path,labels)
    distribuer(split_ratio,padding_path,sub_dirs,redimension_train_path,redimension_validation_path)
    # clean_wav(args.read_path)
    # cut_padding_audio(args.padding_path,sub_dirs,args.T_total,labels,args.redimension_train_path,args.redimension_validation_path)
    # clean_wav(args.padding_path)


def extract(read_path,redimension_train_path,redimension_validation_path,mfcc_length,
                    sample_size,sample_rate,length,width,image_train_path,image_validation_path):
    print(redimension_train_path)
    print(mfcc_length)
    ############ lstm mfcc
    cmd = "rm -rf feature/*"
    sh.run(cmd)
    file_ext='*.wav'
    # sub_dirs,labels = get_spices(read_path)
    sub_dirs = ['Bm','Eg']
    print(sub_dirs)
    lstm_train_features_mfcc,lstm_train_labels_mfcc = parse_audio_files_mfcc(redimension_train_path,sub_dirs,file_ext,mfcc_length)
    lstm_validation_features_mfcc,lstm_validation_labels_mfcc = parse_audio_files_mfcc(redimension_validation_path,sub_dirs,file_ext,mfcc_length)

    np.savetxt("feature/lstm_train_features_mfcc.txt",lstm_train_features_mfcc)
    np.savetxt("feature/lstm_train_labels_mfcc.txt",lstm_train_labels_mfcc)
    np.savetxt("feature/lstm_validation_features_mfcc.txt",lstm_validation_features_mfcc)
    np.savetxt("feature/lstm_validation_labels_mfcc.txt",lstm_validation_labels_mfcc)
    ############ lstm wavelet

    lstm_train_features_wavelet,lstm_train_labels_wavelet = parse_audio_files_waveletPackets(redimension_train_path,sub_dirs,file_ext)
    lstm_validation_features_wavelet,lstm_validation_labels_wavelet = parse_audio_files_waveletPackets(redimension_validation_path,sub_dirs,file_ext)

    np.savetxt("feature/lstm_train_features_wavelet.txt",lstm_train_features_wavelet)
    np.savetxt("feature/lstm_train_labels_wavelet.txt",lstm_train_labels_wavelet)
    np.savetxt("feature/lstm_validation_features_wavelet.txt",lstm_validation_features_wavelet)
    np.savetxt("feature/lstm_validation_labels_wavelet.txt",lstm_validation_labels_wavelet)
    ###########  lstm raw signal
    # sample_size=200
    # sample_rate=100
    lstm_train_features_rawSignal,lstm_train_labels_rawSignal = parse_audio_files_rawSignal(redimension_train_path,sub_dirs,file_ext,sample_size,sample_rate)
    lstm_validation_features_rawSignal,lstm_validation_labels_rawSignal = parse_audio_files_rawSignal(redimension_train_path,sub_dirs,file_ext,sample_size,sample_rate)

    np.savetxt("feature/lstm_train_features_rawSignal.txt",lstm_train_features_rawSignal)
    np.savetxt("feature/lstm_train_labels_rawSignal.txt",lstm_train_labels_rawSignal)
    np.savetxt("feature/lstm_validation_features_rawSignal.txt",lstm_validation_features_rawSignal)
    np.savetxt("feature/lstm_validation_labels_rawSignal.txt",lstm_validation_labels_rawSignal)

    ################   cnn mfcc
    # length = 28
    # width = 28
    cnn_train_features_mfcc,cnn_train_labels_mfcc = get_cnn_mfccs(redimension_train_path,sub_dirs,file_ext,length,width)
    cnn_validation_features_mfcc,cnn_validation_labels_mfcc = get_cnn_mfccs(redimension_validation_path,sub_dirs,file_ext,length,width)
    # Write the array to disk
    with file('feature/cnn_train_features_mfcc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(cnn_train_features_mfcc.shape))
        for data_slice in cnn_train_features_mfcc:
            np.savetxt(outfile, data_slice, fmt='%-7.6f')
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    np.savetxt("feature/cnn_train_labels_mfcc.txt",cnn_train_labels_mfcc)

    with file('feature/cnn_validation_features_mfcc.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(cnn_validation_features_mfcc.shape))
        for data_slice in cnn_validation_features_mfcc:
            np.savetxt(outfile, data_slice, fmt='%-7.6f')
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    np.savetxt("feature/cnn_validation_labels_mfcc.txt",cnn_validation_labels_mfcc)

    ############ cnn raw signal
    clean_image(image_train_path)
    clean_image(image_validation_path)
    rawSignal_to_image(redimension_train_path,sub_dirs,file_ext,image_train_path)
    rawSignal_to_image(redimension_validation_path,sub_dirs,file_ext,image_validation_path)

    ########### wavenet mfcc
    net,rec_fields,max_size = wavenet()

    wavenet_train_features_mfcc,wavenet_train_labels_mfcc = parse_audio_files_mfcc(redimension_train_path,sub_dirs,file_ext,max_size)
    wavenet_validation_features_mfcc,wavenet_validation_labels_mfcc = parse_audio_files_mfcc(redimension_validation_path,sub_dirs,file_ext,max_size)

    np.savetxt("feature/wavenet_train_features_mfcc.txt",wavenet_train_features_mfcc)
    np.savetxt("feature/wavenet_train_labels_mfcc.txt",wavenet_train_labels_mfcc)
    np.savetxt("feature/wavenet_validation_features_mfcc.txt",wavenet_validation_features_mfcc)
    np.savetxt("feature/wavenet_validation_labels_mfcc.txt",wavenet_validation_labels_mfcc)

    ########### wavenet raw signal
    wavenet_train_features_rawSignal,wavenet_train_labels_rawSignal = parse_audio_files_rawSignal(redimension_train_path,
                                                            sub_dirs,file_ext,max_size,sample_rate)
    wavenet_validation_features_rawSignal,wavenet_validation_labels_rawSignal = parse_audio_files_rawSignal(redimension_validation_path,
                                                            sub_dirs,file_ext,max_size,sample_rate)

    np.savetxt("feature/wavenet_train_features_rawSignal.txt",wavenet_train_features_rawSignal)
    np.savetxt("feature/wavenet_train_labels_rawSignal.txt",wavenet_train_labels_rawSignal)
    np.savetxt("feature/wavenet_validation_features_rawSignal.txt",wavenet_validation_features_rawSignal)
    np.savetxt("feature/wavenet_validation_labels_rawSignal.txt",wavenet_validation_labels_rawSignal)


def train_model(features_type,arc,hidden_size,num_layers,num_classes,drop_out,lr,batch_size,epochs,
                    split_ratio,length,width,optimizer,image_train_path,image_validation_path):

    if features_type == 'mfcc' and arc =='lstm':
        # train_features,train_labels = parse_audio_files_mfcc(redimension_train_path,sub_dirs,file_ext,mfcc_length)
        # validation_features,validation_labels = parse_audio_files_mfcc(redimension_validation_path,sub_dirs,file_ext,mfcc_length)
        print("begin to train mfcc by lstm")
        train_features = np.loadtxt("feature/lstm_train_features_mfcc.txt")
        train_labels = np.loadtxt("feature/lstm_train_labels_mfcc.txt")
        validation_features = np.loadtxt("feature/lstm_validation_features_mfcc.txt")
        validation_labels = np.loadtxt("feature/lstm_validation_labels_mfcc.txt")

        input_size = train_features.shape[1]
        # print("input size is ",input_size)
        model = RNN(input_size,hidden_size,num_layers, num_classes,drop_out)

        # define optimizer and loss function
        optim,loss_func = optimizer_lossFunc(optimizer,model,lr)


        loss,model,accuracy,precision,recall,f1,auc,training_time = train_rnn(train_features,train_labels,validation_features,validation_labels,model,
                                            optim,loss_func,input_size,batch_size,epochs)

        torch.save(model, 'model/mfcc_model_lstm.pkl')
        # print type(training_time)
        print ("training time " ,training_time)
        write_result(loss,accuracy,precision,recall,f1,auc,training_time,features_type,epochs,batch_size,split_ratio,arc)

    elif features_type == 'wavelet'and arc =='lstm':
        # train_features,train_labels = parse_audio_files_waveletPackets(args.redimension_train_path,sub_dirs,file_ext)
        # validation_features,validation_labels = parse_audio_files_waveletPackets(args.redimension_validation_path,sub_dirs,file_ext)
        train_features = np.loadtxt("feature/lstm_train_features_wavelet.txt")
        train_labels = np.loadtxt("feature/lstm_train_labels_wavelet.txt")
        validation_features = np.loadtxt("feature/lstm_validation_features_wavelet.txt")
        validation_labels = np.loadtxt("feature/lstm_validation_labels_wavelet.txt")

        input_size = train_features.shape[1]
        model = RNN(input_size,hidden_size, num_layers, num_classes,drop_out)

        # define optimizer and loss function
        optim,loss_func = optimizer_lossFunc(optimizer,model,lr)


        loss,model,accuracy,precision,recall,f1,auc,training_time = train_rnn(train_features,train_labels,validation_features,validation_labels,model,
                                            optim,loss_func,input_size,batch_size,epochs)

        torch.save(model, 'model/wavelet_model_lstm.pkl')
        print ("training time " ,training_time)
        write_result(loss,accuracy,precision,recall,f1,auc,training_time,features_type,epochs,batch_size,split_ratio,arc)

    elif features_type == 'raw_signal'and arc =='lstm':
        # train_features,train_labels = parse_audio_files_rawSignal(args.redimension_train_path,sub_dirs,file_ext,args.sample_size,args.sample_rate)
        # validation_features,validation_labels = parse_audio_files_rawSignal(args.redimension_train_path,sub_dirs,file_ext,args.sample_size,args.sample_rate
        train_features = np.loadtxt("feature/lstm_train_features_rawSignal.txt")
        train_labels = np.loadtxt("feature/lstm_train_labels_rawSignal.txt")
        validation_features = np.loadtxt("feature/lstm_validation_features_rawSignal.txt")
        validation_labels = np.loadtxt("feature/lstm_validation_labels_rawSignal.txt")

        input_size = train_features.shape[1]
        # print input_size
        model = RNN(input_size,hidden_size, num_layers, num_classes,drop_out)

        # define optimizer and loss function
        optim,loss_func = optimizer_lossFunc(optimizer,model,lr)


        loss,model,accuracy,precision,recall,f1,auc,training_time = train_rnn(train_features,train_labels,validation_features,validation_labels,model,
                                            optim,loss_func,input_size,batch_size,epochs)
        torch.save(model, 'model/rawSignal_model_lstm.pkl')
        print ("training time " ,training_time)
        write_result(loss,accuracy,precision,recall,f1,auc,training_time,features_type,epochs,batch_size,split_ratio,arc)

    elif features_type == 'mfcc'and arc =='cnn':
        # train_features,train_labels = get_cnn_mfccs(args.redimension_train_path,sub_dirs,file_ext,args.length,args.width)
        # validation_features,validation_labels = get_cnn_mfccs(args.redimension_validation_path,sub_dirs,file_ext,args.length,args.width)
        train_features = np.loadtxt("feature/cnn_train_features_mfcc.txt")
        print train_features.shape[0]
        train_features = train_features.reshape(train_features.shape[0]/length,length,width)
        train_labels = np.loadtxt("feature/cnn_train_labels_mfcc.txt")

        validation_features = np.loadtxt("feature/cnn_validation_features_mfcc.txt")
        validation_features = validation_features.reshape(validation_features.shape[0]/length,length,width)
        validation_labels = np.loadtxt("feature/cnn_validation_labels_mfcc.txt")


        in_channels = 1
        model = CNN(num_classes,drop_out,length,width,in_channels)

        # define optimizer and loss function
        optim,loss_func = optimizer_lossFunc(optimizer,model,lr)


        loss,training_time,model,accuracy,precision,recall,f1,auc = train_mfcc(train_features,train_labels,validation_features,
                                                        validation_labels,model,optim,loss_func,batch_size,epochs)

        torch.save(model, 'model/mfcc_model_cnn.pkl')
        print ("training time " ,training_time)
        write_result(loss,accuracy,precision,recall,f1,auc,training_time,features_type,epochs,batch_size,split_ratio,arc)

    elif features_type == 'raw_signal'and arc =='cnn':
        # clean_image(args.image_train_path)
        # clean_image(args.image_validation_path)
        # rawSignal_to_image(args.redimension_train_path,sub_dirs,file_ext,args.image_train_path)
        # rawSignal_to_image(args.redimension_validation_path,sub_dirs,file_ext,args.image_validation_path)

        in_channels = 3
        model = CNN(num_classes,drop_out,length,width,in_channels)

        # define optimizer and loss function
        optim,loss_func = optimizer_lossFunc(optimizer,model,lr)

        loss,training_time  = train_rawSignal(image_train_path,model,
                                            optim,loss_func,batch_size,epochs,length,width)
        _,model,accuracy,precision,recall,f1,auc = predict_rawSignal(image_validation_path,model,batch_size,length,width)
        torch.save(model, 'model/rawSignal_model_cnn.pkl')
        print ("training time " ,training_time)
        write_result(loss,accuracy,precision,recall,f1,auc,training_time,features_type,epochs,batch_size,split_ratio,arc)


    elif features_type == 'mfcc'and arc =='wavenet':
        ########### wavenet mfcc
        model,rec_fields,_ = wavenet()

        # wavenet_train_features_mfcc,wavenet_train_labels_mfcc = parse_audio_files_mfcc(args.redimension_train_path,sub_dirs,file_ext,max_size)
        # wavenet_validation_features_mfcc,wavenet_validation_labels_mfcc = parse_audio_files_mfcc(args.redimension_validation_path,sub_dirs,file_ext,max_size)

        train_features = np.loadtxt("feature/wavenet_train_features_mfcc.txt")
        train_labels = np.loadtxt("feature/wavenet_train_labels_mfcc.txt")
        validation_features = np.loadtxt("feature/wavenet_validation_features_mfcc.txt")
        validation_labels = np.loadtxt("feature/wavenet_validation_labels_mfcc.txt")

        # define optimizer and loss function
        optim,loss_func = optimizer_lossFunc(optimizer,model,lr)

        loss,accuracy,precision,recall,f1,auc,training_time = train_wavenet(model,rec_fields,train_features,train_labels,validation_features,validation_labels,
                                                optim,loss_func,batch_size,epochs)
        print ("training time " ,training_time)
        write_result(loss,accuracy,precision,recall,f1,auc,training_time,features_type,epochs,batch_size,split_ratio,arc)

    elif features_type == 'raw_signal'and arc =='wavenet':
        ########### wavenet raw signal
        # wavenet_train_features_rawSignal,wavenet_train_labels_rawSignal = parse_audio_files_rawSignal(args.redimension_train_path,
                                                                # sub_dirs,file_ext,max_size,args.sample_rate)
        # wavenet_validation_features_rawSignal,wavenet_validation_labels_rawSignal = parse_audio_files_rawSignal(args.redimension_validation_path,
                                                                # sub_dirs,file_ext,max_size,args.sample_rate)

        train_features = np.loadtxt("feature/wavenet_train_features_rawSignal.txt")
        train_labels = np.loadtxt("feature/wavenet_train_labels_rawSignal.txt")
        validation_features = np.loadtxt("feature/wavenet_validation_features_rawSignal.txt")
        validation_labels = np.loadtxt("feature/wavenet_validation_labels_rawSignal.txt")

        model,rec_fields,_ = wavenet()

        # define optimizer and loss function
        optim,loss_func = optimizer_lossFunc(optimizer,model,lr)

        loss,accuracy,precision,recall,f1,auc,training_time  = train_wavenet(model,rec_fields,train_features,train_labels,validation_features,validation_labels,
                                                optim,loss_func,batch_size,epochs)
        print ("training time " ,training_time)
        write_result(loss,accuracy,precision,recall,f1,auc,training_time,features_type,epochs,batch_size,split_ratio,arc)

def optimizer_lossFunc(optimizer,model,lr):

    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    return optimizer,loss_func

def evaluate(prediction_labels,pred_y):
    # print prediction_labels
    # print pred_y
    print ("Classification Report:")
    print (metrics.classification_report(prediction_labels, pred_y))
    print ("Confusion Matrix:")
    print(metrics.confusion_matrix(prediction_labels, pred_y))

    print ("Accuracy: {:.2f}%".format(100* accuracy_score(prediction_labels, pred_y)))
    print ("precision: {:.2f}%".format(100 * precision_score(prediction_labels, pred_y, average='macro')))
    print ("recall: {:.2f}%".format(100 * recall_score(prediction_labels, pred_y, average='macro')))
    print ("f1 score: {:.2f}%".format(100 * f1_score(prediction_labels, pred_y, average='macro')))
    print ("auc area: {:.2f}%".format(100 * roc_auc_score(prediction_labels, pred_y)))
    # print ("training time %.4f s" % training_time)
    accuracy = "{:.2f}%".format(100* accuracy_score(prediction_labels, pred_y))
    # print accuracy
    precision = "{:.2f}%".format(100* precision_score(prediction_labels, pred_y))
    recall = "{:.2f}%".format(100* recall_score(prediction_labels, pred_y))
    f1 = "{:.2f}%".format(100* f1_score(prediction_labels, pred_y))
    auc = "{:.2f}%".format(100* roc_auc_score(prediction_labels, pred_y))
    # training_time = "{:.4f} s".format(training_time)
    return accuracy,precision,recall,f1,auc

def write_result(loss,accuracy,precision,recall,f1,auc,training_time,features_type,epochs,batch_size,split_ratio,arc):
    csvFile = open("result.csv","w")
    writer = csv.writer(csvFile)
    dataset = ["Dataset","DCLDE"]
    test_type = ["Test D/C","D/C"]
    labels_methode = ["Learning Presentation",features_type.upper()]
    num_epochs = ["Number Epochs",epochs]
    batch = ["batch size",batch_size]
    ratio = ["train/test split",split_ratio]
    cross_validation = ["cross validation","Hold-out method"]
    classifier = ["Classifier",arc.upper()]
    loss = ["loss",loss]
    accuracy = ["accuracy",accuracy]
    precision = ["precision",precision]
    recall = ["recall",recall]
    f1 = ["f1",f1]
    auc = ["auc",auc]
    training_time = ["training_time",training_time]
    writer.writerow(dataset)
    writer.writerow(test_type)
    writer.writerow(labels_methode)
    writer.writerow(num_epochs)
    writer.writerow(batch)
    writer.writerow(ratio)
    writer.writerow(cross_validation)
    writer.writerow(classifier)
    writer.writerow(loss)
    writer.writerow(accuracy)
    writer.writerow(precision)
    writer.writerow(recall)
    writer.writerow(f1)
    writer.writerow(auc)
    writer.writerow(training_time)
    csvFile.close()

'''
def split_data(features,labels,split_ratio):
    # random train and test sets.
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size = split_ratio, random_state=0)

    #print np.random.rand(len(train_features))

    # train_test_split = np.random.rand(len(train_features)) < split_ratio
    # #print train_test_split
    # train_x = train_features[train_test_split]
    # train_y = train_labels[train_test_split]
    # test_x = train_features[~train_test_split]
    # test_y = train_labels[~train_test_split]
    #
    #
    # #print train_x.shape,train_y.

    return train_x,train_y,test_x,test_y
'''

def clean_wav(file_dir):
    for i, f in enumerate(glob.glob(file_dir + os.sep +'*')):
        # print f
        cmd = "rm -rf " + f  + "/*.wav"
        sh.run(cmd)

def clean_image(file_dir):
    for i, f in enumerate(glob.glob(file_dir + os.sep +'*')):
        # print f
        cmd = "rm -rf " + f  + "/*.png"
        sh.run(cmd)

def splitext(wav_name):
    name = wav_name.split(".")

    return name



def plot_spec(audio_path):
    y, sr = librosa.load(audio_path, duration=10)
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
    librosa.display.specshow(D, y_axis='log',x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    # plt.tight_layout()
    plt.show()

def plot_wave(audio_path):
    y, sr = librosa.load(audio_path)
    y_harm, y_perc = librosa.effects.hpss(y)
    plt.figure(figsize=(10, 4))
    librosa.display.waveplot(y_harm, sr=sr, alpha=0.25)
    librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5)
    plt.title('Harmonic + Percussive')
    plt.show()

if __name__ == '__main__':
    # output_dir = "result_redimension"
    # file_directory_name = "bm_redimension"
    # waves,names = get_files(output_dir,file_directory_name)
    #
    # raw_waves = raw_sounds = load_sound_files(waves)
    # # plot_specgram(names, raw_waves)
    # plot_waves(names, raw_waves)
    audio_path = 'wav/train/WAT_HZ_01_160217_221115.df100.x.wav'
    # audio_path = 'combine.wav'
    plot_spec(audio_path)
    # plot_wave(audio_path)
