import sys
import numpy as np
import os
import csv
import glob
import pandas as pd
import sh
from datetime import datetime, timedelta, tzinfo

class FixedOffset(tzinfo):
    """offset_str: Fixed offset in str: e.g. '-0400'"""
    def __init__(self, offset_str):
        sign, hours, minutes = offset_str[0], offset_str[1:3], offset_str[3:]
        offset = (int(hours) * 60 + int(minutes)) * (-1 if sign == "-" else 1)
        self.__offset = timedelta(minutes=offset)
        # NOTE: the last part is to remind about deprecated POSIX GMT+h timezones
        # that have the opposite sign in the name;
        # the corresponding numeric value is not used e.g., no minutes
        '<%+03d%02d>%+d' % (int(hours), int(minutes), int(hours)*-1)
    def utcoffset(self, dt=None):
        return self.__offset
    def tzname(self, dt=None):
        return self.__name
    def dst(self, dt=None):
        return timedelta(0)
    def __repr__(self):
        return 'FixedOffset(%d)' % (self.utcoffset().total_seconds() / 60)

def read(sample_file):

    with open(sample_file, 'rb') as csvfile:

    	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        segProjet = []
        segSite = []

        segStart = []
        #segEnd = []
        segLabel = []
        duration = []
        segQuality = []
        for j, row in enumerate(reader):
            segProjet.append(row[0])
            segSite.append(row[1])
            date_with_tz = row[3]
            #print date_with_tz

            start = datetime.strptime(date_with_tz, "%Y-%m-%dT%H:%M:%S.%f")

            #start = datetime.datetime.strptime(row[3], "%Y-%m-%dT%H:%M:%S")
            segStart.append(start)
            #segEnd.append(row[4])
            end = datetime.strptime(row[4], "%Y-%m-%dT%H:%M:%S.%f")
            segLabel.append(row[2])
            dur = (end - start).total_seconds()
            duration.append(dur)

            segQuality.append(row[5])
            #print dur
    #print segProjet,segSite,segStart,duration,segLabel,segQuality

    return segProjet,segSite,segStart,duration,segLabel,segQuality


def date_type(wav_dir,segProjet,segSite,segStart,duration,segLabel,segQuality,result_bm_dir,result_eg_dir):
    print max(duration)
    print min(duration)
    for i, f in enumerate(glob.glob(wav_dir + os.sep +'*.wav')):               # for each WAV file
        wavFile = f
        #print os.path.splitext(wavFile)[0]
        waveFile_name = (os.path.splitext(wavFile)[0]).split(os.sep)[1]
        #print waveFile_name
        date1 = waveFile_name.split("_")[3]
        date2 = (waveFile_name.split("_")[4]).split(".")[0]
        #print date1,date2
        temp1 = []
        temp2 = []
        for i in range(len(date1)):
            #print date1[i], '(%d)' %i
            temp1.append(date1[i])
        date1 = "20" + temp1[0] + temp1[1] +"-"+ temp1[2] + temp1[3] +"-"+ temp1[4]+temp1[5]
        print ("date1 is ",date1)

        for j in range(len(date2)):
            #print date2[j], '(%d)' %j
            temp2.append(date2[j])
        date2 = temp2[0] + temp2[1] +":"+ temp2[2] + temp2[3] +":"+ temp2[4]+temp2[5]
        #print date2

        date_str = date1 +" "+ date2
        #print data_str

        start_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

        print ("start_date is ",start_date)
        # cut_time_start = []
        # cut_time_duration = []
        # cut_time_label = []
        # for k in range(len(segStart)):
        #     start_csv = segStart[k]
        #     #print start_csv
        #     start_csv1 = start_csv.strftime("%Y-%m-%d %H:%M:%S.%f").split(" ")
        #     print start_csv1[0]
        #     if start_csv1[0] == date1:
        #         print("~~~~~~~~~~~~~~~~~")
        #         cut_time_start.append(segStart[k])
        #         cut_time_duration.append(duration[k])
        #         cut_time_label.append(segLabel[k])
        #
        # print  len(cut_time_start),len(cut_time_duration),len(cut_time_label)

        cut(wavFile, segProjet,segSite,segStart,duration,segLabel,segQuality, result_bm_dir,result_eg_dir,start_date)




def cut(wavFile,segProjet,segSite,segStart,duration,segLabel,segQuality, result_bm_dir,result_eg_dir,start_date):
    # date2 = datetime.strptime(date2, "%H:%M:%S")
    #print("date2 is ",date2)
    cmd = """ffmpeg -i """ + wavFile + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
    rs = sh.run(cmd, True)

    duration_in_wavFile = rs.stdout()
    print("the duration of wavfile is ", duration_in_wavFile)
    print type(duration_in_wavFile)
    print type(duration)
    good_projet = []
    good_site = []
    good_species = []
    good_start = []
    good_cut_point = []
    good_duration = []
    good_quality = []
    for i in range(0, len(segStart)):
        if i < len(segStart) - 1:
            cut_point = (segStart[i] - start_date).total_seconds()
            #print cut_point
            #print type(cut_point)

            if cut_point >= 0 and float(cut_point) <= float(duration_in_wavFile):
                good_projet.append(segProjet[i])
                good_site.append(segSite[i])
                good_species.append(segLabel[i])

                temp_start = segStart[i].strftime("%Y-%m-%dT%H:%M:%S.%f")
                temp_start = temp_start[:-5]
                good_start.append(temp_start)

                good_cut_point.append(cut_point)
                good_duration.append(duration[i])
                good_quality.append(segQuality[i])
    print good_cut_point,str(good_duration)
    print good_start
    for j in range(0,len(good_cut_point)):
        str_name = str(good_projet[j]) +"_"+ str(good_site[j]) +"_"+ str(good_start[j]) +"."+ str(good_species[j]) +"."+ str(good_quality[j])
        if str(good_species[j]) == "Bm":
            cmd = "ffmpeg -ss " + str(good_cut_point[j]) + " -t " + str(good_duration[j])+ " -i " + wavFile + " " + result_bm_dir + "/" + str_name + ".wav"
            sh.run(cmd)
        if str(good_species[j]) == "Eg":
            cmd = "ffmpeg -ss " + str(good_cut_point[j]) + " -t " + str(good_duration[j])+ " -i " + wavFile + " " + result_eg_dir + "/" + str_name + ".wav"
            sh.run(cmd)

    # resultFile = "result/0.wav"
    # cmd = """ffmpeg -i """ + resultFile + """ 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | awk '{ split($1, A, ":"); print 3600*A[1] + 60*A[2] + A[3] }'"""
    # rs = sh.run(cmd, True)
    # duration_in_resultFile = rs.stdout()
    # print("the duration of resultfile is ", duration_in_resultFile)




def combine(result_dir,combine_dir):
    part = ""
    size = 0
    for i, f in enumerate(glob.glob(result_dir + os.sep +'*.wav')):
        part += " -i " + f
        size +=1
    #print part
    cmd = "ffmpeg " + part + " -filter_complex '[0:0][1:0][2:0][3:0]concat=n=" + str(size) + ":v=0:a=1[out]' -map '[out]' " + combine_dir + "/combine.wav"
    sh.run(cmd)



if __name__ == '__main__':

    sample_file = "sample/HAT_A_LF_dev.csv"

    wav_dir = "wav"

    result_bm_dir = "read/read_bm"
    result_eg_dir = "read/read_eg"
    combine_dir = "combine_wavFile"
    #clean files
    cmd = "rm -rf " + result_bm_dir  + "/*"
    sh.run(cmd)
    cmd = "rm -rf " + result_eg_dir  + "/*"
    sh.run(cmd)
    # cmd = "rm -rf " + combine_dir  + "/*"
    # sh.run(cmd)

    segProjet,segSite,segStart,duration,segLabel,segQuality = read(sample_file)
    date_type(wav_dir,segProjet,segSite,segStart,duration,segLabel,segQuality,result_bm_dir,result_eg_dir)
    # combine(result_dir,combine_dir)



    # date_with_tz = "2017-01-12T14:12:06.000-0500"
    # date_str, tz = date_with_tz[:-5], date_with_tz[-5:]
    # print date_str,tz
    # dt_utc = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
    # dt = dt_utc.replace(tzinfo=FixedOffset(tz))
    # print(dt)
