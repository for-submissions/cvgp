
import numpy as np

from sklearn.utils import shuffle
from sklearn.impute import KNNImputer
from sklearn.datasets import make_moons

from uci_datasets import Dataset

def load_dataset(
        dataset='synthetic',
        ):

    if 'synthetic' in dataset:
        data = np.load('./data/{}/observed.npy'.format(dataset))
        # data = data
        y = data[:,0]
        X = data[:,1:]
        dtypes = np.asarray([X.dtype] * X.shape[1])
    elif 'moons' in dataset:
        X, y = make_moons(n_samples=10000, noise=0.05, random_state=0)
        dtypes = np.asarray([X.dtype] * X.shape[1])
    else:
        data = Dataset(dataset)
        x_train, y_train, x_test, y_test = data.get_split(split=0)
        X = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test]).reshape(-1)
        y = (y - y.mean()) / (y.std() + 1e-5)
        imputer = KNNImputer(n_neighbors=2)
        X = imputer.fit_transform(X)
        std = (X.std(0) < 1e-5) #if std is too small by-pass as there can be numerical errors
        X = X[:, ~std]
        uniques = [len(np.unique(x, axis=0)) for x in X.T]
        uniques_ = [len(np.unique(x, axis=0)) for x in X.astype(np.int64).T]
        dtypes = [u == u_ for (u, u_) in zip(uniques, uniques_)]
        dtypes_ = [u_ <= 5 for u_ in uniques_]
        dtypes = np.asarray(dtypes) * np.asarray(dtypes_) * (
            np.sum(X.astype(np.int64).T - X.T, 1) == 0
            )
        dtypes = np.asarray(
            [
                np.dtype(
                    np.float64
                    ) if not d else np.dtype(
                        np.int64
                        ) for d in dtypes
                        ]
                        )

    return shuffle(X, y), dtypes
