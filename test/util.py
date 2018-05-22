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


AUDIO_EXT = "wav"
AUDIO_NUM = 3

def splitext(wav_name):
    name = wav_name.split(".")

    return name



def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds


def get_files(output_dir,file_directory_name):
    file_directory_path = os.path.join(output_dir, file_directory_name)
    files = os.listdir(file_directory_path)
    waves = []
    names = []
    for file in files:

        if AUDIO_EXT in file:
            names.append(file)
            waves.append(os.path.join(os.path.join(output_dir, file_directory_name, file)))
            if len(waves) >= AUDIO_NUM:
                break
    # print waves
    print names
    return waves,names

def plot_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=100)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(3, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Spectrogram', x=0.5, y=0.985, fontsize=18)
    #plt.show()
    plt.savefig(os.path.join("result_redimension", 'f1_spec.jpg'))



def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25, 60), dpi=100)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(3, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot', x=0.5, y=0.985, fontsize=18)
    #plt.show()
    plt.savefig(os.path.join("result_redimension", 'f1_waveplot.jpg'))


if __name__ == '__main__':
    output_dir = "result_redimension"
    file_directory_name = "original"
    waves,names = get_files(output_dir,file_directory_name)

    raw_waves = raw_sounds = load_sound_files(waves)
    # plot_specgram(names, raw_waves)
    plot_waves(names, raw_waves)
