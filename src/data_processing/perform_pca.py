import os, re, sys
import numpy as np
from scipy import linalg
from read_data import read_preprocessed_data
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score



FEATURE_PATH = '../../data/aaindex/aaindex_used.txt'
FEATURE_TO_FIT = 123
FEATURE_KEPT = 20

def train_pca(Xtr):
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    #X_scaled = preprocessing.scale(min_max_scaler.fit_transform(np.array(Xtr)))
    X_scaled = preprocessing.scale(np.array(Xtr))
    pca = PCA(svd_solver='auto', n_components=FEATURE_TO_FIT)
    pca.fit(X_scaled)
    #print "Variance ration among dimensions: ", pca.explained_variance_ratio_
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
    #save_transfromed_train_data(Ytr, Xtr, pca, label_socres_path=output_path)
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

    #Y, X = create_data(input_file)
    #pca_preprocess(Y, X, data_folder)

    pass