from random import shuffle
from sklearn.model_selection import KFold


class SplitData:

    def __init__(self, Y, X, n_splits):
        self.n_splits = n_splits
        self.Xtr = []; self.Xte = []
        self.Ytr = []; self.Yte = []

        zipped = zip(Y, X)
        shuffle(zipped)
        Y, X = zip(*zipped)
        kf = KFold(n_splits=self.n_splits)
        for train_idx, test_idx in kf.split(X):
            self.Xtr.append([X[i] for i in train_idx])
            self.Xte.append([X[i] for i in test_idx])
            self.Ytr.append([Y[i] for i in train_idx])
            self.Yte.append([Y[i] for i in test_idx])
        pass

