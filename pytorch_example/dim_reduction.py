import os
import csv
import json
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler



def get_pca(features):
    pca = PCA(n_components=2)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed)


def get_scaled_tsne_embeddings(features, perplexity, iteration):
    embedding = TSNE(n_components=2,
                     perplexity=perplexity,
                     n_iter=iteration).fit_transform(features)
    scaler = MinMaxScaler()
    scaler.fit(embedding)
    return scaler.transform(embedding)


def transform_numpy_to_json(array):
    data = []
    for position in array:
        data.append({
            'coordinates': position.tolist()
        })
    return data

def read_mfcc():
    read_dir = 'read/'
    sub_dirs = ['Ba', 'Bm','Eg']
    file_ext='*.wav'
    dataset = []
    errors = 0

    sample_rate = 44100
    mfcc_size = 13
    for label, sub_dir in enumerate(sub_dirs):
        print("label: %s" % (label))
        #print("sub_dir: %s" % (sub_dir))
        for f in glob.glob(os.path.join(read_dir, sub_dir, file_ext)):
            print(f)
            try:
                data, _ = librosa.load(f)

                trimmed_data, _ = librosa.effects.trim(y=data)

                mfccs = librosa.feature.mfcc(trimmed_data,
                                             sample_rate,
                                             n_mfcc=mfcc_size)

                stddev_mfccs = np.std(mfccs, axis=1)

                mean_mfccs = np.mean(mfccs, axis=1)

                average_difference = np.zeros((mfcc_size,))
                for i in range(0, len(mfccs.T) - 2, 2):
                    average_difference += mfccs.T[i] - mfccs.T[i+1]
                average_difference /= (len(mfccs) // 2)
                average_difference = np.array(average_difference)

                concat_features = np.hstack((stddev_mfccs, mean_mfccs))
                concat_features = np.hstack((concat_features, average_difference))


                dataset += [(f, concat_features)]

            except:
                print("error!")
                errors += 1
    # print(dataset)
    print('errors:', errors)
    return dataset

if __name__ == "__main__":

    dataset = read_mfcc()

    all_file_paths, mfcc_features = zip(*dataset)

    mfcc_features = np.array(mfcc_features)

    mfcc_tuples = []

    all_json = dict()
    '''
    all_json["filenames"] = all_file_paths

    print(len(all_file_paths),
          mfcc_features.shape)
    '''
    #####t-sne
    tsne_embeddings_mfccs = []
    perplexities = [2, 5, 10]
    iterations = [300, 500]
    for i, perplexity in enumerate(perplexities):
        for j, iteration in enumerate(iterations):
            tsne_mfccs = get_scaled_tsne_embeddings(mfcc_features,
                                                    perplexity,
                                                    iteration)

            tsne_embeddings_mfccs.append(tsne_mfccs)

            mfcc_key = 'tsnemfcc{}{}'.format(i, j)

            all_json[mfcc_key] = transform_numpy_to_json(tsne_mfccs)


    fig, ax = plt.subplots(nrows=len(perplexities),
                           ncols=len(iterations),
                           figsize=(30, 30))

    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            current_plot = i * len(iterations) + j
            col.scatter(tsne_embeddings_mfccs[current_plot].T[0],
                        tsne_embeddings_mfccs[current_plot].T[1],
                        s=1)
    plt.show()

    #### PCA
    pca_mfcc = get_pca(mfcc_features)
    plt.figure(figsize=(30, 30))
    _ = plt.scatter(pca_mfcc.T[0],
                    pca_mfcc.T[1])
    plt.show()

    ### stocker
    json_name = "data.json"
    json_string = "d = '" + json.dumps(all_json) + "'"
    with open(json_name, 'w') as json_file:
        json_file.write(json_string)
