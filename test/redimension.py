import numpy as np
from sys import argv
import sys
import os
import glob
import sh
import matplotlib.pyplot as plt
import librosa



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




def read_audio(wav_dir,T_total):
    for i, f in enumerate(glob.glob(wav_dir + os.sep +'*.wav')):               # for each WAV file

        wavFile = f
        print wavFile
        # Read the wav file
        in_data, fs = librosa.load(wavFile)

        print("fs is ",fs)
        print("data is ", in_data)
        #print os.path.splitext(wavFile)[0]
        waveFile_name = (os.path.splitext(wavFile)[0]).split(os.sep)[1]

        cmd = """ffmpeg -i """ + wavFile + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
        rs = sh.run(cmd, True)

        duration_in_wavFile = rs.stdout()
        print("the duration of wavfile is ", duration_in_wavFile)
        print type(duration_in_wavFile)

        result_padding_dir = "result_padding"
        if float(T_total) >= float(duration_in_wavFile):
            out_data = padding_audio(in_data,fs,T_total)
            # Save the output file

            out_wav = result_padding_dir+"/"+waveFile_name+".wav"
            librosa.output.write_wav(out_wav, out_data, fs)

        else:
            int_num = int(float(duration_in_wavFile) / T_total)+1
            number = float(duration_in_wavFile) % T_total
            print number,int_num
            if number != 0:
                T = T_total*int_num
                out_data = padding_audio(in_data,fs,T)
                out_wav = result_padding_dir+"/"+waveFile_name+".wav"
                librosa.output.write_wav(out_wav, out_data, fs)

            # cut_point = np.linspace(0,40,11)
            # print cut_point
            # for j in range(0,number):
            #     if j < number - 1:
            #         cmd = "ffmpeg -ss " + str(cut_point[j]) + " -t " + str(T_total)+ " -i " + wavFile + " " + result_dir + "/" + waveFile_name +"_"+ str(j) + ".wav"
            #         sh.run(cmd)
            #     else:
            #         temp_dur = float(duration_in_wavFile) - T_total*(number-1)
            #         if float(T_total) >= float(temp_dur):
            #             out_data = padding_audio(in_data,fs,T_total)
            #             # Save the output file
            #
            #             out_wav = result_dir+"/"+waveFile_name+".wav"
            #             wavf.write(out_wav, fs, out_data






if __name__ == "__main__":

    wav_dir = "result_eg"
    #define total duration is 8s
    T_total = 4
    cmd = "rm -rf " + "result_padding"  + "/*"
    sh.run(cmd)

    read_audio(wav_dir,T_total)
