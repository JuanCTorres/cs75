from sklearn import svm
from sklearn.model_selection import cross_val_score
import sys
sys.path.append('/Users/imac/Documents/CS75/liblinear/python')
sys.path.append('../')
from liblinearutil import *
from data_processing.read_data import read_preprocessed_data

INPUT_FILE = '../../data/plants/label_scores.txt'
FEATURES_FILE = '../../data/aaindex/aaindex_used.txt'



def set_svm(labels, features):
    clf = svm.SVC(kernel='linear', cache_size=500)
    clf.fit(features, labels)
    return clf
    pass

# -s svm_type : set type of SVM (default 0)
#     0 -- C-SVC
#     1 -- nu-SVC
#     2 -- one-class SVM
#     3 -- epsilon-SVR
#     4 -- nu-SVR
# -t kernel_type : set type of kernel function (default 2)
#     0 -- linear: u'*v
#     1 -- polynomial: (gamma*u'*v + coef0)^degree
#     2 -- radial basis function: exp(-gamma*|u-v|^2)
#     3 -- sigmoid: tanh(gamma*u'*v + coef0)
# -d degree : set degree in kernel function (default 3)
# -g gamma : set gamma in kernel function (default 1/num_features)
# -r coef0 : set coef0 in kernel function (default 0)
# -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
# -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
# -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
# -m cachesize : set cache memory size in MB (default 100)
# -e epsilon : set tolerance of termination criterion (default 0.001)
# -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
# -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
# -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)

def main():
    Y, X = read_preprocessed_data(INPUT_FILE, FEATURES_FILE)
    name_dict = dict()
    for label in Y:
        name_dict.setdefault(label, len(name_dict))
    int_y = []
    for label in Y:
        int_y.append(name_dict[label])
        pass

    print 'labels: ', len(Y) , len(X), len(X[0])
    prob = problem(int_y, X)
    param = parameter('-C -s 0 -v 5')
    m = train(prob, param)

    #clf = svm.SVC(kernel='linear', cache_size=1000, decision_function_shape = 'ovr')
    #scores = cross_val_score(clf, X[0:150], Y[0:150], cv=5)
    #print scores
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    pass

if __name__ == '__main__':
    main()