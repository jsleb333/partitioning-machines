import sys, os
sys.path.append(os.getcwd())

import numpy as np
from graal_utils import Timer
from sklearn.metrics import accuracy_score

from experiments.models import OursShaweTaylorPruning
from experiments.datasets import Wine
from experiments.cross_validator import CrossValidator

n_folds = 5
seed = 57
n_draws = 10

dataset = Wine(shuffle=seed)
model = OursShaweTaylorPruning(max_n_leaves=75, error_prior_exponent=1)
model.fit_tree(dataset)
exponents = np.linspace(1, 20, num=20)
cv = CrossValidator(dataset, model, n_folds)

def func_to_maximize(dtc, X_test, y_test, param):
    dtc.error_prior_exponent = param
    dtc._prune_tree(dataset)
    return accuracy_score(y_pred=dtc.predict(X_test), y_true=y_test)

best_exponents = {e:0 for e in exponents}

for draw in range(n_draws):
    with Timer(f'Draw #{draw}'):
        draw_exponents = cv.cross_validate(func_to_maximize, exponents, seed+draw*10, verbose=False, return_best_param='all')
        for fold_exponents in draw_exponents:
            for e in fold_exponents:
                best_exponents[e] += 1
values = np.array(list(best_exponents.values()))
total = sum(values)
mean = sum(k*v for k, v in best_exponents.items())/total
modes = [k for k, v in best_exponents.items() if v == np.max(values)]
cumul = 0
for k, v in best_exponents.items():
    cumul += v
    if cumul > total/2:
        break
    median = k


print(mean, modes, f'({np.mean(modes)})', median)
# Prints 11.448275862068966 [8.0, 9.0, 10.0, 11.0] (9.5) 11.0
print(best_exponents)
