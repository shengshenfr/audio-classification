import numpy as np
import sh
import sys
import os
import csv
import glob

def main():
    # mp3_dir = "mp3"
    feature_dir = "feature"
    model_dir = "model"
    prediction_wav_dir = "prediction_wav"
    read_dir = "read"
    padding_dir = "padding"
    redimension_dir = "redimension"
    window_dir = "window"
    sample_csv_dir = "csvData"
    wav_dir = "wavData"
    image_dir = "image"
    dir_name = feature_dir +" "+ model_dir +" "+ prediction_wav_dir +" "+ read_dir +" "+ padding_dir+" "+redimension_dir \
                +" "+ sample_csv_dir  +" "+ wav_dir+" "+ window_dir+" "+image_dir
    sh.run("mkdir -p "+ dir_name)
    #sh.run("rm -rf "+str_name)

    # mp3_to_wav.run(mp3_file, wav_dir)
    wav_train__dir = wav_dir + "/" + "train"
    wav_prediction_dir = wav_dir + "/" + "prediction"
    csv_train_dir = sample_csv_dir + "/" + "train"
    csv_prediction_dir = sample_csv_dir + "/" + "prediction"

    redim_train_dir = redimension_dir+ "/" + "train"
    redim_validation_dir = redimension_dir+ "/" +"validation"
    # padding_train_dir = padding_dir+ "/" + "train"
    # padding_validation_dir = padding_dir+ "/" + "validation"
    # padding_test_dir = padding_dir+ "/" + "test"

    image_train_dir = image_dir+ "/" + "train"
    image_validation_dir = image_dir+ "/" + "validation"
    # image_test_dir = image_dir+ "/" + "test"

    sub_dir = wav_train__dir +" "+ wav_prediction_dir +" "+ csv_train_dir +" "+ csv_prediction_dir +" "+redim_train_dir \
                            +" "+redim_validation_dir+" "+image_train_dir+" "+image_validation_dir \

    sh.run("mkdir -p "+ sub_dir)

    labels = []
    for i, f in enumerate(glob.glob(sample_csv_dir + os.sep +'*')):
        # print f
        for j,f_sub in enumerate(glob.glob(f + os.sep +'*.csv')):
            # print f_sub
            sample_file = f_sub
            with open(sample_file, 'rb') as csvfile:

            	reader = csv.reader(csvfile, delimiter=',', quotechar='|')

                for k, row in enumerate(reader):
                    labels.append(row[2])


    print np.unique(labels)
    for l in np.unique(labels):
        print l
        sh.run("mkdir -p "+ read_dir +"/"+ l)
        sh.run("mkdir -p "+ padding_dir +"/"+ l)
        # sh.run("mkdir -p "+ redimension_dir +"/"+ l)
        sh.run("mkdir -p "+ redim_train_dir +"/"+ l)
        sh.run("mkdir -p "+ redim_validation_dir +"/"+ l)
        sh.run("mkdir -p "+ prediction_wav_dir +"/"+ l)

        # sh.run("mkdir -p "+ padding_train_dir +"/"+ l)
        # sh.run("mkdir -p "+ padding_validation_dir +"/"+ l)
        # sh.run("mkdir -p "+ padding_test_dir +"/"+ l)
        sh.run("mkdir -p "+ image_train_dir +"/"+ l)
        sh.run("mkdir -p "+ image_validation_dir +"/"+ l)
        # sh.run("mkdir -p "+ image_test_dir +"/"+ l)

if __name__ == '__main__':
	main()
