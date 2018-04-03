
import sys
import numpy as np
import os
import sklearn.svm
import csv
import glob
from sklearn import preprocessing 

def read_csv_file(csvFile):

    with open(csvFile, 'rb') as csvfile:
    	reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    	#T_period = [row[0] for row in reader]
    	features = []
    	result_class = []
    	totalNumOfFeatures = 4
        for j, row in enumerate(reader):
    		#T1 = row[0]
    		if j==0:
    			continue
    		#BW1 = row[1]
    		else:
    			x = np.zeros((totalNumOfFeatures, 1))
    			x[0] = row[0]
    			x[1] = row[1]
    			x[2] = row[2]
    			x[3] = row[3]

    			y = np.zeros((1, 1))
    			y[0] = row[4]
    			# BW.append(row[1])
    			# centroid.append(row[2])
    			# Fmax.append(row[3])
    			# lable.append(row[4])
    			features.append(x)
    			result_class.append(y)

    	#print(T,BW,centroid,Fmax,lable)	
    return features,result_class



def randSplitFeatures(features, partTrain):

    featuresTrain = []
    featuresTest = []
    for i, f in enumerate(features):
        [numOfSamples, numOfDims] = f.shape
        randperm = numpy.random.permutation(range(numOfSamples))
        nTrainSamples = int(round(partTrain * numOfSamples))
        featuresTrain.append(f[randperm[0:nTrainSamples]])
        featuresTest.append(f[randperm[nTrainSamples::]])
    return (featuresTrain, featuresTest)





def trainSVM(features, Cparam):


    [X, Y] = listOfFeatures2Matrix(features)
    svm = sklearn.svm.SVC(C = Cparam, kernel = 'linear',  probability = True)        
    svm.fit(X,Y)

    return svm

def normalize_features(features):
	X = np.array([])
	for count,f in enumerate(features):
		if f.shape[0] > 0:
			if count == 0:
				X = f

			else:
				#print f
				X = np.vstack((X,f))
			count += 1

	mean = np.mean(X, axis = 0) + 0.00000000000001;
	std = np.std(X,axis = 0) + 0.00000000000001;

	features_normalization = []
	for f in features:
		ft = f.copy()
		for n_sample in range(f.shape[0]):
			#print f
			#print(f.shape[0])
			print ft[n_sample, :]
			ft[n_sample, :] = (ft[n_sample, :] - mean)/ std
		features_normalization.append(ft)
	return (features_normalization, mean, std)	

def evaluate_classfier(features,class_names,nExp,classifier_name,params,parameter_mode,per_train = 0.70):
	(features_normalization, mean, std) = normalize_features(features)



def feature_train(features,result_class):
	num_features = len(features)
	#print(" number features",num_features)



if __name__ == '__main__':  
	csvFile = "./TableFeats.csv" 




	###  step 1 : read csv file


	features,class_name= read_csv_file(csvFile)
	# features = numpy.concatenate(features, 1)
	#print(features)
	numOfFeatures = features[0].T.shape[1]
	print(numOfFeatures)
	# features 1 = T, features 1 = Bw, features 1 = Fcentroid, features 1 = Fmax
	featureNames = ["features" + str(d + 1) for d in range(numOfFeatures)]
	#print featureNames

	for i,f in enumerate(features):
		if len(f) == 0:
			print "train_svm_feature error"	+ listofDir[i]

	'''
	choose parameter
	'''
 	
	classfierParams = np.array([0.001, 0.01, 0.5, 1.0, 5.0, 10.0, 20.0])
 	print classfierParams

 	### get optimal class parameter
 	features1 = []
 	for f in features:
 		f_temp = []
 		for i in range(f.shape[0]):
 			temp = f[i,:]
 			if (not np.isnan(temp).any()) and (not np.isinf(temp).any()):
 				f_temp.append(temp.tolist())
 				
 			else:
 				print "feature error"

 		features1.append(np.array(f_temp))
 	#print features2							

 	features = features1


 	features2 = []
 	features2 = normalize_features(features)
 	features = features2

 	#print features2

 	#best_parameter = evaluate_classifier(features,)
	# features = preprocessing.minmax_scale(features,axis=0,feature_range=(0,1))
	
	# print(features)
	# result_class = numpy.concatenate(result_class, 1)	
	# #print result_class
	# feature_train(features,result_class)