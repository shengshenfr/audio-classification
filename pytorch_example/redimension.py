import numpy as np
import sys
import os
import glob
import sh
import matplotlib.pyplot as plt
import librosa
import util
import random
import shutil

def padding_audio(data,fs,T):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape
    N_pad = N_tar - shape[0]
    print("Padding with %f seconds of silence" % float(N_pad/fs))
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append
    if shape[0] > 0:
        if len(shape) > 1:
            return np.vstack((data,np.zeros(shape)))
        else:
            return np.hstack((data,np.zeros(shape)))
    else:
        return data




def read_audio(read_dir,sub_dirs,T_total,padding_dir,labels):
    for k, sub_dir in enumerate(sub_dirs):
        #print("label: %s" % (label))
        print("sub_dir: %s" % (sub_dir))
        for i, f in enumerate(glob.glob(read_dir + os.sep+ sub_dir + os.sep +'*.wav')):               # for each WAV file

            wavFile = f
            print wavFile
            waveFile_name = (os.path.splitext(wavFile)[0]).split(os.sep)[2]
            print waveFile_name
            quality = util.splitext(waveFile_name)[2]
            print quality
            # Read the wav file
            in_data, fs = librosa.load(wavFile)

            #print("fs is ",fs)
            #print("data is ", in_data)
            #print os.path.splitext(wavFile)[0]


            cmd = """ffmpeg -i """ + wavFile + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
            rs = sh.run(cmd, True)

            duration_in_wavFile = rs.stdout()
            print("the duration of wavfile is ", duration_in_wavFile)
            print type(duration_in_wavFile)

            if float(T_total) >= float(duration_in_wavFile):
                out_data = padding_audio(in_data,fs,T_total)

                # Save the output file
                for l in labels:
                    if quality == l:
                        out_wav = padding_dir +"/"+ l +"/"+ waveFile_name +".wav"
                    # if quality == "Eg":
                    #     out_wav = result_padding_dir +"/"+ eg_padding_dir +"/" +waveFile_name+".wav"

                    librosa.output.write_wav(out_wav, out_data, fs)


            else:
                int_num = int(float(duration_in_wavFile) / T_total)+1
                number = float(duration_in_wavFile) % T_total
                print number,int_num
                if number != 0:
                    T = T_total*int_num
                    out_data = padding_audio(in_data,fs,T)
                    for l in labels:
                        if quality == l:
                            out_wav = padding_dir +"/"+ l +"/"+ waveFile_name +".wav"
                        # if quality == "Eg":
                        #     out_wav = result_padding_dir +"/"+ eg_padding_dir +"/" +waveFile_name+".wav"

                        librosa.output.write_wav(out_wav, out_data, fs)


def distribuer(ratio,padding_dir,sub_dirs,redimension_train_dir,redimension_validation_dir):
    for k, sub_dir in enumerate(sub_dirs):
        for i, f in enumerate(glob.glob(padding_dir + os.sep+ sub_dir + os.sep)):
            # print(f)
            name_list=list(os.path.join(f,name) for name in os.listdir(f))

            print(len(name_list))
            numfiles = int(round(len(name_list)*ratio))
            name_list = np.random.permutation(name_list)
            # random_name_list=list(random.sample(name_list,numfiles))
            # print(name_list)
            # print(len(random_name_list))
            for i in range(numfiles):
                # print(oldname)
                # print(name_list[i])
                specie = name_list[i].split("/")[1]
                # print(specie)
                dstPath = redimension_train_dir+"/"+specie+"/"
                shutil.copyfile(name_list[i],name_list[i].replace(f, dstPath))

            for i in range(numfiles,len(name_list)):
                # print(oldname)
                # print(name_list[i])
                specie = name_list[i].split("/")[1]
                # print(specie)
                dstPath = redimension_validation_dir+"/"+specie+"/"
                shutil.copyfile(name_list[i],name_list[i].replace(f, dstPath))
'''


def cut_padding_audio(padding_dir,sub_dirs,T_total,labels,redimension_train_dir,redimension_prediction_dir):
    for k, sub_padding in enumerate(sub_dirs):
        #print("label: %s" % (label))
        print("sub_padding: %s" % (sub_padding))
        for i, f in enumerate(glob.glob(padding_dir + os.sep + sub_padding + os.sep+'*.wav')):               # for each WAV file

            wavFile = f
            print wavFile
            waveFile_name = (os.path.splitext(wavFile)[0]).split(os.sep)[2]
            quality = util.splitext(waveFile_name)[2]
            print quality

            cmd = """ffmpeg -i """ + wavFile + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
            rs = sh.run(cmd, True)

            duration_in_wavFile = rs.stdout()
            print("the duration of wavfile is ", duration_in_wavFile)
            int_num = int(float(duration_in_wavFile) / float(T_total))-1
            print int_num
            cut_point = np.linspace(0,T_total*int_num,int_num+1)
            print cut_point
            if float(duration_in_wavFile) > float(T_total):
                for j in range(0, len(cut_point)):
                    print str(cut_point[j])
                    for l in labels:

                        if quality == l:
                            number = np.random.randint(0,9)
                            if number>2:
                                path_name = redimension_train_dir +"/"+ l +"/"
                            else:
                                path_name = redimension_prediction_dir +"/"+ l +"/"
                        # if quality == "Eg":
                        #     path_name = result_redimension_dir +"/"+ eg_redimension_dir +"/"

                    cmd = "ffmpeg -ss " + str(cut_point[j]) + " -t " + str(T_total)+ " -i " + wavFile + " " + path_name + waveFile_name +"_"+ str(j) + ".wav"
                    sh.run(cmd)
            else:
                for l in labels:
                    if quality == l:
                        number = np.random.randint(0,9)
                        if number>2:
                            path_name = redimension_train_dir +"/"+ l +"/"
                        else:
                            path_name = redimension_prediction_dir +"/"+ l +"/"
                    # if quality == "Eg":
                    #     path_name = result_redimension_dir +"/"+ eg_redimension_dir +"/"

                cmd = "cp " + wavFile +" "+ path_name
                sh.run(cmd)

'''



if __name__ == "__main__":

    # wav_dir = "result_eg"
    read_dir = "read"
    padding_dir = "padding"
    # redimension_dir = "redimension"
    redimension_train_dir = "redimension/train"
    redimension_validation_dir = "redimension/validation"
    # padding_train_dir = "padding/train"
    # padding_validation_dir = "padding/validation"
    # padding_test_dir = "padding/test"
    labels = []
    T_total = 40
    sub_dirs = []
    for i, f in enumerate(glob.glob(read_dir + os.sep +'*')):
        f = os.path.splitext(f)[0]
        sub_dirs.append(f.split(os.sep)[1])
        labels.append(f.split(os.sep)[1])
    print labels

    util.clean_wav(redimension_train_dir)
    util.clean_wav(redimension_validation_dir)
    # util.clean_wav(padding_test_dir)
    read_audio(read_dir,sub_dirs,T_total,padding_dir,labels)
    ratio = 0.7
    distribuer(ratio,padding_dir,sub_dirs,redimension_train_dir,redimension_validation_dir)
