from typing import Callable, Any
from numbers import Number
import numpy as np
from sklearn.model_selection import KFold
from copy import copy

import sys, os
sys.path.append(os.getcwd())

from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion


class CrossValidator:
    def __init__(self, dataset, model, n_folds=10) -> None:
        self.dataset = dataset
        self.model = model
        self.n_folds = n_folds

    def cross_validate(
            self,
            func_to_maximize: Callable[[DecisionTreeClassifier, np.ndarray, np.ndarray, Any], Number],
            param_to_optimize: list,
            seed=37
        ):
        fold_idx = list(KFold(n_splits=self.n_folds,
                              shuffle=True,
                              random_state=seed).split(self.dataset.X_train))

        best_params = []
        for fold, (tr_idx, ts_idx) in enumerate(fold_idx):
            X_train, y_train = self.dataset.X_train[tr_idx], self.dataset.y_train[tr_idx]
            X_test, y_test = self.dataset.X_train[ts_idx], self.dataset.y_train[ts_idx]
            dtc = copy(self.model).fit(X_train, y_train, nominal_mask=self.model.nominal_mask)

            objectives = []
            for param in param_to_optimize:
                copy_of_dtc = copy(dtc)
                copy_of_dtc.tree = copy(dtc.tree)
                objectives.append(func_to_maximize(dtc, X_test, y_test, param))

            best_params.append(param_to_optimize[np.argmax(objectives)])

        return best_params
