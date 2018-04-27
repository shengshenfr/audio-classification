import numpy as np
from sys import argv
import sys
import os
import glob
import sh
import matplotlib.pyplot as plt
import librosa
import util





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
            return np.vstack((np.zeros(shape),
                              data))
        else:
            return np.hstack((np.zeros(shape),
                              data))
    else:
        return data




def read_audio(read_dir,sub_dirs,T_total,result_padding_dir,bm_padding_dir,eg_padding_dir):
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
                if quality == "Bm":
                    out_wav = result_padding_dir +"/"+ bm_padding_dir +"/"+ waveFile_name +".wav"
                if quality == "Eg":
                    out_wav = result_padding_dir +"/"+ eg_padding_dir +"/" +waveFile_name+".wav"

                librosa.output.write_wav(out_wav, out_data, fs)


            else:
                int_num = int(float(duration_in_wavFile) / T_total)+1
                number = float(duration_in_wavFile) % T_total
                print number,int_num
                if number != 0:
                    T = T_total*int_num
                    out_data = padding_audio(in_data,fs,T)

                    if quality == "Bm":
                        out_wav = result_padding_dir +"/"+ bm_padding_dir +"/"+ waveFile_name +".wav"
                    if quality == "Eg":
                        out_wav = result_padding_dir +"/"+ eg_padding_dir +"/" +waveFile_name+".wav"

                    librosa.output.write_wav(out_wav, out_data, fs)




def cut_padding_audio(result_padding_dir,sub_paddings,T_total,result_redimension_dir,bm_redimension_dir,eg_redimension_dir):
    for k, sub_padding in enumerate(sub_paddings):
        #print("label: %s" % (label))
        print("sub_padding: %s" % (sub_padding))
        for i, f in enumerate(glob.glob(result_padding_dir + os.sep + sub_padding + os.sep+'*.wav')):               # for each WAV file

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
                    if quality == "Bm":
                        path_name = result_redimension_dir +"/"+ bm_redimension_dir +"/"
                    if quality == "Eg":
                        path_name = result_redimension_dir +"/"+ eg_redimension_dir +"/"

                    cmd = "ffmpeg -ss " + str(cut_point[j]) + " -t " + str(T_total)+ " -i " + wavFile + " " + path_name + waveFile_name +"_"+ str(j) + ".wav"
                    sh.run(cmd)
            else:
                if quality == "Bm":
                    path_name = result_redimension_dir +"/"+ bm_redimension_dir +"/"
                if quality == "Eg":
                    path_name = result_redimension_dir +"/"+ eg_redimension_dir +"/"

                cmd = "cp " + wavFile +" "+ path_name
                sh.run(cmd)





if __name__ == "__main__":

    # wav_dir = "result_eg"
    read_dir = "read"


    result_bm_padding_dir = "result_padding/bm_padding"
    result_eg_padding_dir = "result_padding/eg_padding"

    result_padding_dir = "result_padding"
    bm_padding_dir = "bm_padding"
    eg_padding_dir = "eg_padding"

    result_redimension_dir = "result_redimension"
    bm_redimension_dir = "bm_redimension"
    eg_redimension_dir = "eg_redimension"
    sub_paddings = ['bm_padding', 'eg_padding']
    #define total duration is 8s
    T_total = 4
    # cmd = "rm -rf " + result_bm_padding_dir  + "/*.wav"
    # sh.run(cmd)
    # cmd = "rm -rf " + result_eg_padding_dir  + "/*.wav"
    # sh.run(cmd)
    sub_dirs = ['read_bm', 'read_eg']
    #read_audio(read_dir,sub_dirs,T_total,result_padding_dir,bm_padding_dir,eg_padding_dir)
    cut_padding_audio(result_padding_dir,sub_paddings,T_total,result_redimension_dir,bm_redimension_dir,eg_redimension_dir)
