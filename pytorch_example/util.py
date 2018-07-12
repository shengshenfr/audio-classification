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

# import multiprocessing
# import GPUtil as GPU

'''
def monitor_gpu():
    usages = []
    while True:
       gpu = GPU.getGPUs()[0]
       usage = gpu.load*100
       usages.append(usage)
       time.sleep(interval)
    print usages
'''

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
    ratio = ["train/test split",1-split_ratio]
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


def split_data(features,labels,split_ratio):
    # random train and test sets.
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size = split_ratio, random_state=0)

    #print np.random.rand(len(train_features))
    '''
    train_test_split = np.random.rand(len(train_features)) < split_ratio
    #print train_test_split
    train_x = train_features[train_test_split]
    train_y = train_labels[train_test_split]
    test_x = train_features[~train_test_split]
    test_y = train_labels[~train_test_split]


    #print train_x.shape,train_y.
    '''
    return train_x,train_y,test_x,test_y


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
