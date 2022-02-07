import sys, os
sys.path.append(os.getcwd())

import numpy as np
from graal_utils import Timer
from sklearn.metrics import accuracy_score

from experiments.models import OursHypInvPruning
from experiments.datasets import Wine
from experiments.cross_validator import CrossValidator
from experiments.utils import geo_mean

n_folds = 5
seed = 37
n_draws = 5

dataset = Wine(shuffle=seed)
model = OursHypInvPruning(pfub_factor=1, max_n_leaves=75)
model.fit_tree(dataset)
pfub_factors = np.logspace(0, 20, num=21, base=10)
cv = CrossValidator(dataset, model, n_folds)

def func_to_maximize(dtc, X_test, y_test, param):
    dtc.pfub_factor = param
    dtc._prune_tree(dataset)
    return accuracy_score(y_pred=dtc.predict(X_test), y_true=y_test)

best_pfub_constants = []

for draw in range(n_draws):
    best_pfub_constants.extend(cv.cross_validate(func_to_maximize, pfub_factors, seed+draw*10, verbose=False))

print(geo_mean(best_pfub_constants), np.log10(geo_mean(best_pfub_constants)))
# Prints 1.9e7
