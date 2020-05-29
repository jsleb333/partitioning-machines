from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
import csv
from time import time

from graal_utils import Timer

from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion, shawe_taylor_bound_pruning_objective_factory
from experiments.pruning import prune_with_bound

from datasets.datasets import load_datasets, dataset_list
from train import train


def launch_experiment(dataset,
                      test_split_ratio=.25,
                      n_draws=25,
                      n_folds=10,
                      max_n_leaves=40,
                      error_prior_exponent=13.1,
                      exp_name=None):
    exp_name = exp_name if exp_name is not None else datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
    
    exp_params = {
        'exp_name':exp_name,
        'test_split_ratio':test_split_ratio,
        'n_draws':n_draws,
        'n_folds':n_folds,
        'max_n_leaves':max_n_leaves,
        'error_prior_exponent':error_prior_exponent,
        }
    
    X, y = dataset.data, dataset.target

    # exp_path = f'./experiments/results/{dataset.name}/test_date/'
    exp_path = f'./experiments/results/{dataset.name}/{exp_name}/'

    os.makedirs(exp_path, exist_ok=True)

    with open(exp_path + 'exp_params.py', 'w') as file:
        file.write(f"exp_params = {exp_params}")

    model_name = 'ours'

    file = open(exp_path + model_name + '.csv', 'w', newline='')
    csv_writer = csv.writer(file)

    header = ['draw', 'seed', 'train_accuracy', 'test_accuracy', 'n_leaves', 'height', 'bound', 'time']

    csv_writer.writerow(header)
    file.flush()

    for draw in range(n_draws):
        print(f'Running draw #{draw}')
        
        seed = draw*10 + 1
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y,
                                                test_size=test_split_ratio,
                                                random_state=seed)

        decision_tree = DecisionTreeClassifier(gini_impurity_criterion, max_n_leaves=max_n_leaves)
        n_examples, n_features = X.shape
        r = 1/2**error_prior_exponent
        errors_logprob_prior = lambda n_err: np.log(1-r) + n_err * np.log(r)
        bound = shawe_taylor_bound_pruning_objective_factory(n_features, errors_logprob_prior=errors_logprob_prior)
    
        decision_tree.fit(X_tr, y_tr)
        print(decision_tree.tree.n_leaves)

        t_start = time()
        decision_tree.bound_value = prune_with_bound(decision_tree, bound)
        elapsed_time = time() - t_start
        print(decision_tree.tree.n_leaves)
        
        acc_tr = accuracy_score(y_tr, decision_tree.predict(X_tr))
        acc_ts = accuracy_score(y_ts, decision_tree.predict(X_ts))
        leaves = decision_tree.tree.n_leaves
        height = decision_tree.tree.height
        bound = decision_tree.bound_value
        csv_writer.writerow([draw, seed, acc_tr, acc_ts, leaves, height, bound, elapsed_time])
        file.flush()

    file.close()


if __name__ == "__main__":
    
    exp_name = 'debug'
    
    # for dataset in load_datasets(['iris', 'wine']):
    datasets = list(load_datasets())
    for dataset in datasets:
        with Timer(f'Dataset {dataset.name}'):
            launch_experiment(dataset,
                              error_prior_exponent=13.7,
                              exp_name=exp_name)