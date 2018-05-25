# -*- coding: utf-8 -*-
"""
@Filename: PyAudioTest
@Date: 2017-oct.-06-15-51
@Poject: tempoDetecction
@Author: paul
@Version: 1.0
@Notes: 
@Usage: 
"""

#from pyAudioAnalysis import audioBasicIO
#from scikits.talkbox.features import mfcc

import pandas as pd
import matplotlib.pyplot as plt
import scipy
from pylab import *
import numpy as np
from matplotlib.mlab import find
import warnings
warnings.filterwarnings("ignore")
from numpy.fft import fft
from scipy.signal import filtfilt, butter
from scipy.io import loadmat
from collections import Iterable
eps = 0.00000000000001
def compute_max_magnitude(serie, LFFT_feature_func, fe_func, window=0):

    gamma_s = 1.0 / (len(serie) * fe_func) * abs(fft(serie, LFFT_feature_func) ** 2)
    freq_range = range(0, LFFT_feature_func)
    freq = asarray([freq_range[ite] * double(fe_func) / LFFT_feature_func for ite in range(0, LFFT_feature_func)])
    indi_int = find(freq <= fe_func / 2.0)
    gamma_s_int_func = gamma_s[indi_int]
    freq_int_func = freq[indi_int]
    indi_maximum_func = argmax(gamma_s_int_func)
    return indi_maximum_func, freq_int_func, gamma_s_int_func

def compute_threshold(filtered_data, Q1, Q2, Pfa_func):
    # Compute threshold for detection
    buff = filtered_data
    percentil_q1 = np.percentile(buff, Q1 * 100)
    mu = np.percentile(buff, 0.5)
    threshold = - percentil_q1 / np.log(1-Q1) * 0.7 - mu

    return threshold -mu


def filtering(extracted_signal , fe, fmax, fmin, Nint):
    low = fmin / (fe / 2)
    high = fmax / (fe / 2)
    B, A = butter(4, [low, high], btype='bandpass')
    filteredSignal = filtfilt(B, A, extracted_signal)
    return filteredSignal


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, basestring):
            for x in flatten(item):
                yield x
        else:
            yield item

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def window_rms1(a, window_size):
  a2 = np.power(a,2)
  return np.sqrt(running_mean(a2,window_size))


def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)

if __name__ == "__main__":

		# Hydrophone gains
    SH = -165
    G = 22
    D = 1

		# Freq to analyze
    fmin = 10.0
    fmax = 50.0

    Q1 = 0.2
    Q2 = 0.4
    Pfa = 1e-6

    Nint = 512
    garde_sample = 5
    garde_sample_filtre = 3 * Nint
    LFFT_feature = 8192
    vec_index_finst_prop = [0.25,  0.5 , 0.75]
    Lwindow_finst_prop = [0.5]
    Lwindow_SCLinst_prop = [0.25]
    Xpourcent_BW = 0.5

    pas = 10 * 60
    tdeb = 1
    tfin = 23 * 60
    fe = 240.0 # sampling rate
		# Loading file
    vec_s = loadmat('signalTestMultiCris.mat')
    a = vec_s.values()
    #print a
    b = a[0].T
    c = b.tolist()
    d = list(flatten(c))
    vec_s = asarray(d)
    print(len(vec_s))	
		# To smooth the signal envelope
    N = 900
    print vec_s
    #slots = range(0,tfin,pas*240)
    #print("slotss :" , slots)


    DESCRIP_SpectralSpread = np.array([])
    DESCRIP_SpectralCentroid = np.array([])
    #DESCRIP_formants = np.array([])
    #DESCRIP_SpectralRolloff = np.array([])
    DESCRIP_ZCR = np.array([])
    DESCRIP_vec_T = np.array([])

    # for k in range(len(slots)-1):
    #     print("je suis  ssss")
    # 	tdeb = slots[k]
    #     tfin = slots[k+1]
    #     ind_deb = floor(tdeb * fe)
    #     ind_fin = floor(tfin * fe)
    #     vec_s = vec_s[tdeb:tfin+1]

	# Filtering the signal
    filteredSignal = filtering(vec_s, fe, fmax, fmin, Nint)
#    quadratureSignal = filteredSignal ** 2
#   b = ones(int(Nint)) / Nint
#  a = 1
 # envelopeSignal = filtfilt(b, a, quadratureSignal)
	#rmsEnvelopeSignal = np.sqrt(envelopeSignal)
    rmsEnvelopeSignal = window_rms1(filteredSignal,N)
    analyzedSignal = rmsEnvelopeSignal
    threshold = compute_threshold(analyzedSignal, Q1, Q2, Pfa)


    indexesDetection = 1 * (threshold <= analyzedSignal)

    indi_deb = 1
    indi_fin = len(indexesDetection) - 1
    for i in range(0, len(indexesDetection) - 1):
    	  if (indexesDetection[indi_deb] != 0):
    		  indi_deb = indi_deb + 1

    for i in range(len(indexesDetection) - 1, 0, -1):
    	  if (indexesDetection[indi_fin] != 0):
    		  indi_fin = indi_fin - 1

    rightIndexesDetec = indexesDetection[indi_deb:indi_fin]

    risingEdges = find(1 * (diff(rightIndexesDetec) > 0))
    fallingEdges = find(1 * (diff(rightIndexesDetec) < 0))

    copiedData = vec_s
    newCopiedData = copiedData[indi_deb:indi_fin]
    updatedFilteredSignal = filteredSignal[indi_deb:indi_fin]

    DESCRIP_Nb_imp_par_sec = len(risingEdges)


    # Loop through acoustic events to compute features.
    for u in range(0, len(risingEdges)):

    	  indi_deb_court = maximum(1, risingEdges[u])
    	  indi_fin_court = minimum(len(newCopiedData),
    							   fallingEdges[u])

    	  s_court = updatedFilteredSignal[indi_deb_court:indi_fin_court + 1]
    	  env_court = sqrt(updatedFilteredSignal[indi_deb_court:indi_fin_court + 1])

    	  indi_maximum, freq_int, gamma_s_int = compute_max_magnitude(s_court, LFFT_feature, fe)


    	  (a,b) = stSpectralCentroidAndSpread(gamma_s_int,fe)
          # period
    	  DESCRIP_vec_T = append(DESCRIP_vec_T,(fallingEdges[u] - risingEdges[u]) * 1. / fe)
          #centroid
    	  DESCRIP_SpectralCentroid = append(DESCRIP_SpectralCentroid, a)
          # spread
    	  DESCRIP_SpectralSpread = append(DESCRIP_SpectralSpread, b)
    	  #DESCRIP_formants = append(DESCRIP_formants, phormants(s_court, fe))
    	  #DESCRIP_SpectralRolloff = append(DESCRIP_SpectralRolloff,stSpectralRollOff(gamma_s_int,0.25,fe))

    print DESCRIP_vec_T
    print DESCRIP_SpectralSpread
    print DESCRIP_SpectralCentroid
    #print DESCRIP_formants
   # print DESCRIP_SpectralRolloff

    #    plt.figure()
    #   plt.plot(rmsEnvelopeSignal)
    #  plt.plot(threshold*ones(len(rmsEnvelopeSignal)),'r')
    # plt.show()








