import sys
import numpy as np
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from scipy.stats import sem
import pandas as pd
from collections import Iterable
import hmmlearn.hmm
import matplotlib.pyplot as plt

import util
# import time

def train_svm(features_train,labels_train, svm_best_parameter,filename):
    clf = svm.SVC(C = svm_best_parameter,  probability = True)
    clf.fit(features_train,labels_train)
    joblib.dump(clf, filename)

def train_knn(features_train,labels_train, knn_best_parameter,filename):

    knn = KNeighborsClassifier(n_neighbors = knn_best_parameter)
    knn.fit(features_train,labels_train)
    joblib.dump(knn, filename)


def train_hmm(startprob, transmat, means, cov,filename):
    hmm = hmmlearn.hmm.GaussianHMM(startprob.shape[0], "diag")            # hmm training

    hmm.startprob_ = startprob
    hmm.transmat_ = transmat
    hmm.means_ = means
    hmm.covars_ = cov

    joblib.dump(hmm, filename)



def get_hmm_parameter(features,labels):
    uLabel = np.unique(labels)
    #print uLable
    nComps = len(uLabel)

    nFeatures = features.shape[1]
    #print nFeatures
    if features.shape[0] < labels.shape[0]:
        print "trainHMM warning: number of short-term feature vectors must be greater or equal to the labels length!"
        labels = labels[0:features.shape[1]]

    # compute prior probabilities:
    startprob = np.zeros((nComps,))
    for i, u in enumerate(uLabel):
        startprob[i] = np.count_nonzero(labels == u)
    startprob = startprob / startprob.sum()                # normalize prior probabilities
    print ("start probobility is",startprob)

    # compute transition matrix:
    transmat = np.zeros((nComps, nComps))
    print (labels.shape)
    print(labels)
    for i in range(labels.shape[0]-1):
        transmat[int((labels.T)[i]), int((labels.T)[i + 1])] += 1
        print(transmat)
    for i in range(nComps):                     # normalize rows of transition matrix
        transmat[i, :] /= transmat[i, :].sum()

    print ("transfer matrix is",transmat)

    means = np.zeros((nComps, nFeatures))
    for i in range(nComps):
        temp_label = np.nonzero(labels == uLabel[i])[0]
        means[i, :] = np.matrix(features[np.nonzero(labels == uLabel[i])[0],:].T.mean(axis=1))
        #print("#####################")

    print ("means is",means)


    cov = np.zeros((nComps, nFeatures))
    for i in range(nComps):
        #cov[i,:,:] = numpy.cov(features[:,numpy.nonzero(labels==uLabels[i])[0]])  # use this lines if HMM using full gaussian distributions are to be used!
        cov[i, :] = np.std(features[np.nonzero(labels == uLabel[i])[0],:].T, axis=1)

    print ("cov is",cov)
    return startprob, transmat, means, cov


def train_evaluate_cross_validation_svm(features,labels):
    classifierParams = np.linspace(1,30,30)
    # print classifierParams
    accuracy = []
    for c,ci in enumerate(classifierParams):                # for each param value
	# print ci
        classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C = ci))
        scores = cross_validation.cross_val_score(classifier_pipeline, features, labels, cv=10,scoring='accuracy')
        # print scores
        # print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))
        accuracy.append(np.mean(scores))

    best_accuracy_index = np.argmax(accuracy)
    print ("all accuracies are",accuracy)
    print ("best parameter index of C is ",best_accuracy_index)
    print ("best parameter C is ",classifierParams[best_accuracy_index])
    '''
    plt.plot(classifierParams, accuracy)
    plt.xlabel('Value of C for SVM')
    plt.ylabel('Cross-Validated Accuracy')
    ax = plt.axes()
    ann = ax.annotate(u"C = 18.0",xy=(18.0,0.996), xytext=(18,0.993),size=10, va="center",ha="center", bbox=dict(boxstyle='sawtooth',fc="w"), arrowprops=dict(arrowstyle="-|>", connectionstyle="angle,rad=0.4",fc='r') )
    plt.show()
    '''
    return classifierParams[best_accuracy_index]


def train_evaluate_cross_validation_knn(features,labels):
    k_range = range(1,  31)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        #print len(features_normalization)
        #print len(lable[0])
        #features_train, features_test, lable_train, lable_test = train_test_split(features_normalization, lable[0], test_size = 0.2, random_state=0)
        classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), knn)
        scores = cross_validation.cross_val_score(classifier_pipeline, features, labels, cv=10,scoring='accuracy')
        # print scores
        # print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))
        k_scores.append(scores.mean())
    print ("all k scores are",k_scores)
    best_k_scores_index = np.argmax(k_scores)
    print ("best K index is ",best_k_scores_index)
    print ("best parameter K is ",k_range[best_k_scores_index])
    '''
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    ax = plt.axes()
    ann = ax.annotate(u"K = 1",xy=(1.1,0.994), xytext=(4,0.994),size=10, va="center",ha="center", bbox=dict(boxstyle='sawtooth',fc="w"), arrowprops=dict(arrowstyle="-|>", connectionstyle="angle,rad=0.4",fc='r') )

    plt.show()
    '''
    return k_range[best_k_scores_index]


def prediction(features_test,labels_test,model_prediction):



    # print ("features test are ",features_test)
    predictions = np.array([])

    # load the model from disk
    clf = joblib.load(model_prediction)

    for i in range(features_test.shape[0]):
        # X_test = scalingFactor.transform(featureMatrix[i, :])
        X_test = [features_test[i,:]]
        # print("x test is", X_test)
        predictions = np.append(predictions, clf.predict(X_test))
        #print predictions
    # print("prediction is",predictions)

    # print("y_test is",labels_test)
    # print ("Classification Report:")
    # print (metrics.classification_report(labels_test, predictions))
    # print ("Confusion Matrix:")
    # print(metrics.confusion_matrix(labels_test, predictions))
    util.evaluate(labels_test, predictions)
