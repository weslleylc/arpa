import time
from multiprocessing import cpu_count

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from arpa.matching import maximum_weighted_matching as mwm
from arpa.shannon_information import shannon


class ARPA(BaseEstimator, TransformerMixin):

    def __init__(self, k=2, costs=None, function=shannon.process_call_corrcoef,
                 precision=10000000, verbose=0, processes=1):
        self.k = k
        self.costs = costs
        self.function = function
        if function == shannon.process_call_mi:
            self.is_symmetric = True
            self.discretize = True
            self.minimize = False
        elif function == shannon.process_call_cmi:
            self.is_symmetric = True
            self.discretize = True
            self.minimize = False
        elif function == shannon.process_call_icmi:
            self.is_symmetric = False
            self.discretize = True
            self.minimize = True
        elif function == shannon.process_call_mi_cov:
            self.is_symmetric = True
            self.discretize = True
            self.minimize = False
        elif function == shannon.process_call_cov:
            self.is_symmetric = True
            self.discretize = False
            self.minimize = False
        elif function == shannon.process_call_corrcoef:
            self.is_symmetric = True
            self.discretize = False
            self.minimize = False
        else:
            raise ValueError("function {} is not supported".format(function))

        self.obj_cost = None
        self.model = None
        self.precision = precision
        self.verbose = verbose
        self.processes = processes if processes != -1 else cpu_count()
        self.cols = []
        self.elapsed_time = None

    # estimator method
    def fit(self, X, y):
        start = time.time()
        if self.k >= X.shape[1] or self.k < 1:
            self.cols = list(range(X.shape[1]))
            return self

        if self.costs is None:
            self.costs = shannon.calculate_information_matrix(matrix=X, target=y, verbose=self.verbose,
                                                              is_symmetric=self.is_symmetric,
                                                              discretize=self.discretize,
                                                              func=self.function, processes=self.processes)

        costs = self.costs if not self.minimize else (np.max(self.costs) - self.costs)
        self.cols, self.obj_cost, self.model = mwm.calculate_features(costs=costs,
                                                                      k=self.k)

        end = time.time()
        self.elapsed_time = end - start
        return self

    def transform(self, X, y=None):
        return X[:, self.cols]

