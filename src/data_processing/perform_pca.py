import os, re, sys
import numpy as np
from scipy import linalg
from random import shuffle
from read_data import read_preprocessed_data
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


LABEL_SCORE_PATH = '../../data/animals'
PLANTS_LABEL_SCORE_PATH = '../../data/plants'
FEATURE_PATH = '../../data/aaindex/aaindex_used.txt'


def performPCA(X,  batch_size, ):
    pass


def create_data(label_scores):
    Y, X = read_preprocessed_data(label_scores, FEATURE_PATH, exclude_labels_less_than=0, format='default')
    zipped = zip(Y, X)
    shuffle(zipped)
    Y, X = zip(*zipped)
    xarray = np.array(X)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    X_scaled = preprocessing.scale(min_max_scaler.fit_transform(xarray))
    #print X_scaled.mean(axis=0), X_scaled.std(axis=0)
    return Y, X_scaled


def preprocess(Y, X):
    pca = PCA(svd_solver='auto', n_components=566)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    pca.n_components = 20
    Xfit = pca.fit_transform(X)
    with open(LABEL_SCORE_PATH+'/label_scores_projected.txt', 'w') as f:
        for y, x in zip(Y, Xfit):
            line = y + '|'+ '|'.join(format(xx, ".5f") for xx in x)
            f.write(line+'\n')
    pass



if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #     dataset = sys.argv[1]
    # else:
    #     sys.exit(1)

    # if dataset == 'animals':
    data_folder = '../../data/animals'
    input_file = '%s/label_scores.txt' % data_folder
    # else:
    #     raise Exception('Please enter a valid dataset to use. Accepted: \'plants\' and \'animals\'')

    Y, X = create_data(input_file)
    preprocess(Y, X)

    pass