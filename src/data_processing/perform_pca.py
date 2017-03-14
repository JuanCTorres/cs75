import os, re, sys
import numpy as np
from scipy import linalg
from random import shuffle
from read_data import read_preprocessed_data
from split_data import SplitData
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


FEATURE_PATH = '../../data/aaindex/aaindex_all.txt'
FEATURE_TO_FIT = 123
FEATURE_KEPT = 2

INPUT_FILE = "../../data/animals/label_scores.txt"

def cross_val_pca(Xtr):
    n_samples, n_features = len(Xtr), len(Xtr[0])
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    X_scaled = preprocessing.scale(min_max_scaler.fit_transform(np.array(Xtr)))
    pca = PCA(svd_solver='full')
    pca.fit(X_scaled)
    print "Variance ration among dimensions: ", pca.explained_variance_ratio_


    n_components = np.arange(0, n_features, 3)
    pca_scores = []
    for n in n_components:
        pca.n_components = n
        score = np.mean(cross_val_score(pca, Xtr, cv=5))
        pca_scores.append(score)
        print n, score
    return pca_scores
    pass



def train_pca(Xtr):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    X_scaled = preprocessing.scale(min_max_scaler.fit_transform(np.array(Xtr)))
    print 'Xtr: ', len(Xtr), 'Xtr[0]', len(Xtr[0])
    #X_scaled = preprocessing.scale(np.array(Xtr))
    FEATURE_TO_FIT = len(Xtr)
    pca = PCA(svd_solver='full', n_components= len(Xtr[0]))
    pca.fit(X_scaled)
    print "Variance ration among dimensions: ", pca.explained_variance_ratio_
    return pca


# def save_transfromed_train_data(Ytr, Xtr, pca, label_socres_path):
#     pca.n_components = FEATURE_KEPT
#     Xfit = pca.fit_transform(Xtr)
#     with open(label_socres_path+'/label_scores_projected.txt', 'w') as f:
#         for y, x in zip(Ytr, Xfit):
#             line = y + '|'+ '|'.join(format(xx, ".5f") for xx in x)
#             f.write(line+'\n')
#     pass


def tranform_data(X, pca):
    pca.n_components = FEATURE_KEPT
    return pca.fit_transform(X)


def pca_preprocess(Ytr, Xtr, output_path):
    pca = train_pca(Xtr)
    # save_transfromed_train_data(Ytr, Xtr, pca, label_socres_path=output_path)
    return pca
    pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = 'animals'

    if dataset == 'animals':
        data_folder = '../../data/animals'
        input_file = '%s/label_scores.txt' % data_folder
    elif dataset == 'plants':
        data_folder = '../../data/plants'
        input_file = '%s/label_scores.txt' % data_folder
    else:
        raise Exception('Please enter a valid dataset to use. Accepted: \'plants\' and \'animals\'')


    # Y, X = create_data(input_file)
    # pca_preprocess(Y, X, data_folder)
    #Y, X = create_data(input_file)
    # pca_preprocess(Y, X, data_folder)
    Y, X = read_preprocessed_data(input_file, FEATURE_PATH, 5)
    train_pca(Xtr=X)
