
import sys
import numpy as np
import os
import sklearn.svm
import csv
import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from scipy.stats import sem
import pandas as pd


def read_csv_file(csvFile,features,lable):

    with open(csvFile, 'rb') as csvfile:

    	reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    	#T_period = [row[0] for row in reader]

    	totalNumOfFeatures = 4


        for j, row in enumerate(reader):
    		#T1 = row[0]
    		if j==0:
    			continue
    		#BW1 = row[1]
    		else:
    			x = np.zeros((1,totalNumOfFeatures))
    			#print x
    			y = np.zeros((1, 1))
    			#print x

    			x[0][0] = row[0]
    			x[0][1] = row[1]
    			x[0][2] = row[2]
    			x[0][3] = row[3]

    			
    			y[0] = row[4]

    			# BW.append(row[1])
    			# centroid.append(row[2])
    			# Fmax.append(row[3])
    			# lable.append(row[4])
    			features.append(x)
    			lable.append(y)

    	#print(T,BW,centroid,Fmax,lable)
    	
    	#print features	
    return features,lable





def normalize_features(features):
    X = np.array([])

    for count, f in enumerate(features):
        #print X
        if count == 0:
            X = f
        else:
            X = np.vstack((X,f))
        count += 1
    #print X


	#print features   
    features_normalization = preprocessing.scale(X)
    #print features_normalization
    return features_normalization

def train_svm(features_normalization,lable, svm_best_parameter):
    features_train, features_test, lable_train, lable_test = train_test_split(features_normalization, lable[0], test_size = 0.2, random_state=0)
    clf = sklearn.svm.SVC(C = svm_best_parameter,  probability = True)     
    clf.fit(features_train,lable_train)

    return clf

def train_knn(features_normalization,lable, knn_best_parameter):
    features_train, features_test, lable_train, lable_test = train_test_split(features_normalization, lable[0], test_size = 0.2, random_state=0)
    knn = KNeighborsClassifier(n_neighbors = knn_best_parameter)
    knn.fit(features_train,lable_train)

    return knn

def train_evaluate_cross_validation_svm(features_normalization,lable):
    classifierParams = np.array([0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0])
    #print len(features_normalization)
    #print len(lable[0])
    #features_train, features_test, lable_train, lable_test = train_test_split(features_normalization, lable[0], test_size = 0.2, random_state=0)
    accuracy = []
    for c,ci in enumerate(classifierParams):                # for each param value
	print ci
        classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C = ci))
        scores = cross_validation.cross_val_score(classifier_pipeline, features_normalization, lable[0], cv=5,scoring='accuracy')
        print scores
        print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))
        accuracy.append(np.mean(scores))
    
    best_accuracy_index = np.argmax(accuracy)
    print accuracy
    print best_accuracy_index 
    return best_accuracy_index  



      
def train_evaluate_cross_validation_knn(features_normalization,lable):
    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        #print len(features_normalization)
        #print len(lable[0])
        #features_train, features_test, lable_train, lable_test = train_test_split(features_normalization, lable[0], test_size = 0.2, random_state=0)
        classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), knn)
        scores = cross_validation.cross_val_score(classifier_pipeline, features_normalization, lable[0], cv=5,scoring='accuracy')
        print scores
        print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))
        k_scores.append(scores.mean())
    print k_scores
    best_k_scores_index = np.argmax(k_scores)
    
    print best_k_scores_index
    return best_k_scores_index


def prediction(file,model_prediction):
    df = pd.read_csv(file, delimiter=";")
    featureMatrix = np.array(df.iloc[:, :4].values).astype(float)
    classMatrix = np.array(df.iloc[:, 4:].values).astype(int)
    
    print featureMatrix
    print classMatrix.T[0]
    
    predictions = np.array([])

    # load the model from disk
    clf = joblib.load(model_prediction)

    for i in range(featureMatrix.shape[0]):
        # X_test = scalingFactor.transform(featureMatrix[i, :])
        X_test = featureMatrix[i,:]
        predictions = append(predictions, clf.predict(X_test))

    print(predictions)


    print ("Classification Report:")
    print (metrics.classification_report(y_test, predictions))
    print ("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, predictions))

if __name__ == '__main__':  
	csvFile = "./TableFeats.csv" 

	features = []
	lable = []

	###  step 1 : read csv file
	features,lable= read_csv_file(csvFile,features,lable)
	lable = np.concatenate(lable, 1)

	print(lable)
	numOfFeatures = features[0].shape[1]
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
 	#print classfierParams

 	### get optimal class parameter
 	features1 = []
 	for f in features:
 		f_temp = []
 		for i in range(f.shape[0]):
 			#print(f.shape[0])
 			#print("f is  ",f)
			temp = f[i,:]
			if (not np.isnan(temp).any()) and (not np.isinf(temp).any()):
				f_temp.append(temp.tolist())
				
			else:
				print "feature error"

 		features1.append(np.array(f_temp))
 	#print features2							

 	features = features1
 	#print features

 	#class_names = ['type_0','type_1','type_2','type_3','type_4' ]
 	classifier_type = "svm"
 	percent_train_test = 0.70
	#evaluate_classfier(features,lable,1,classifier_type,classfierParams,0,percent_train_test)
 	features_normalization = normalize_features(features)
    	#train_svm(features_normalization,lable)
	#train_knn(features_normalization,lable)
 	# features2 = []
 	# features2 = normalize_features(features)
 	# features = features2

 	#print features2

 	knn_best_parameter = train_evaluate_cross_validation_knn(features_normalization,lable)
    	svm_best_parameter = train_evaluate_cross_validation_svm(features_normalization,lable)


	knn = train_knn(features_normalization,lable, knn_best_parameter)
	svm = train_svm(features_normalization,lable, svm_best_parameter)
	# features = preprocessing.minmax_scale(features,axis=0,feature_range=(0,1))
	
	# print(features)
	# result_class = numpy.concatenate(result_class, 1)	
	# #print result_class
	# feature_train(features,result_class)


	filename = 'finalized_model_knn.sav'
        joblib.dump(knn, filename)

	filename = 'finalized_model_svm.sav'
        joblib.dump(svm, filename)
     

	file = "./featsMat.csv"
	model_prediction = './finalized_model_svm.sav' 
	prediction(file,model_prediction)
