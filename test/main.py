import numpy as np
import mp3_to_wav
import sh
import sys
import os
import csv
import glob
import pandas as pd

def main():
    # mp3_dir = "mp3"

    feature_dir = "feature"
    model_dir = "model"
    prediction_wav_dir = "prediction_wav"
    read_dir = "read"
    resullt_padding_dir = "result_padding"
    result_redimension_dir = "result_redimension"
    sample_csv_dir = "sample_csv"
    wav_dir = "wav"
    dir_name = feature_dir +" "+ model_dir +" "+ prediction_wav_dir +" "+ read_dir +" "+ resullt_padding_dir +" "+ result_redimension_dir +" "+ sample_csv_dir  +" "+ wav_dir
    sh.run("mkdir -p "+ dir_name)
    #sh.run("rm -rf "+str_name)

    # mp3_to_wav.run(mp3_file, wav_dir)

    train_sub_dir = wav_dir + "/" + "train"
    prediction_sub_dir = wav_dir + "/" + "prediction"
    noLabelWav_sub_dir = wav_dir + "/" + "noLabelWav"
    HAT_sub_dir = sample_csv_dir + "/" + "HAT"
    WAT_sub_dir = sample_csv_dir + "/" + "WAT"
    sub_dir = train_sub_dir +" "+ prediction_sub_dir +" "+ noLabelWav_sub_dir +" "+ HAT_sub_dir +" "+ WAT_sub_dir
    sh.run("mkdir -p "+ sub_dir)

    for i, f in enumerate(glob.glob(sample_csv_dir + os.sep +'*')):
        print f

    # with open(sample_file, 'rb') as csvfile:
    #
    # 	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #     Label = []
    #     for j, row in enumerate(reader):
    #         Label.append(row[2])






if __name__ == '__main__':
	main()
