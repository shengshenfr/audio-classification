from python_speech_features import mfcc
from python_speech_features import logfbank

import scipy.io.wavfile as wav
import numpy as np
import sh
import matplotlib.pyplot as plt
import os





def run(mp3_file, wav_dir):
    print(mp3_file)
    cmd = """ffmpeg -i """ + mp3_file + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
    rs = sh.run(cmd, True)
    duration_in_s = rs.stdout()
    print(duration_in_s)
    print("total duration is " + duration_in_s + "s")
    step_s = 600

    for i in range(0, int(float(duration_in_s)), step_s):
        wav_file = wav_dir + "/" + str(i) + ".wav"
        cmd = "ffmpeg -ss " + str(i) + " -t " + str(step_s) + " -i " + mp3_file + " -f wav " + wav_file
        sh.run(cmd)
        print("convert mp3 to wav finish, " + str(i))

        # clean files
        # cmd = "rm -rf " + wav_dir  + "/*"
        # sh.run(cmd)
