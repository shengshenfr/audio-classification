import sys
import os
import glob
import sh
import os.path
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa.display
from scipy import signal

from sklearn import preprocessing

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
