from typing import Callable, Any, Union
from numbers import Number
import numpy as np
from sklearn.model_selection import KFold
from copy import copy
from graal_utils import Timer

import sys, os
sys.path.append(os.getcwd())

from experiments.models import DecisionTreeClassifier


class CrossValidator:
    def __init__(self, dataset, model, n_folds=10) -> None:
        self.dataset = dataset
        self.model = model
        self.n_folds = n_folds

    def cross_validate(
            self,
            func_to_maximize: Callable[[DecisionTreeClassifier, np.ndarray, np.ndarray, Any], Number],
            param_to_optimize: list,
            seed=37,
            verbose=False,
            return_best_param='first',
        ) -> list[Any]:
        """Cross validate a parameter by optimizing a function.

        Args:
            func_to_maximize (Callable[[DecisionTreeClassifier, np.ndarray, np.ndarray, Any], Number]):
                Function used to quantify the performance of the model pruned with given parameter. The first argument is a DecisionTreeClassifier of decision tree, the second and third are the training examples and labels and the last is the parameter to be cross-validated.
            param_to_optimize (list):
                A list of parameter to be cross-validated.
            seed (int, optional):
                A random number generator seed. Defaults to 37.
            verbose (bool, optional):
                If True, will print information about the cross validation process. Defaults to False.
            return_best_param (str, optional):
                Can be either 'first' or 'all'. Defaults to 'first'. If 'first', the first parameter in the list 'param_to_optimize' that achieves the maximum is returned. Otherwise, the list of all parameters in 'param_to_optimize' that achieves the maximum is returned.

        Returns:
            list[Any]:
                By default, returns a list of the parameters that optimizes the function 'func_to_maximize' (one for each fold). If 'return_best_param' is set to 'all', a list of list of parameters is returned instead (one list per fold).
        """
        fold_idx = list(KFold(n_splits=self.n_folds,
                              shuffle=True,
                              random_state=seed).split(self.dataset.X_train))

        iterator = Timer(enumerate(fold_idx)) if verbose else enumerate(fold_idx)
        best_params = []
        for fold, (tr_idx, ts_idx) in iterator:
            if verbose:
                print(f'Proceeding with fold #{fold}.')
            X_train, y_train = self.dataset.X_train[tr_idx], self.dataset.y_train[tr_idx]
            X_test, y_test = self.dataset.X_train[ts_idx], self.dataset.y_train[ts_idx]
            dtc = copy(self.model).fit(X_train, y_train, nominal_mask=self.model.nominal_mask)

            objectives = []
            for param in param_to_optimize:
                copy_of_dtc = copy(dtc)
                copy_of_dtc.tree = copy(dtc.tree)
                objectives.append(func_to_maximize(copy_of_dtc, X_test, y_test, param))

            best_params.append(
                [p for p, o in zip(param_to_optimize, objectives) if o == np.max(objectives)]
                )
            # best_params.append(param_to_optimize[np.argmax(objectives)])
            if verbose:
                print('Objectives:', objectives, '\n')
        if return_best_param == 'first':
            return [b[0] for b in best_params]
        else:
            return best_params

