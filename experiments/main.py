from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
import csv

from graal_utils import Timer

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

    model_names = ['original_tree',
                   'vapnik_tree',
                   'breiman_tree',
                   'modified_breiman_tree',
                   ]

    files = [open(exp_path + name + '.csv', 'w', newline='') for name in model_names]
    csv_writers = [csv.writer(file) for file in files]

    header = ['draw', 'seed', 'train_accuracy', 'test_accuracy', 'n_leaves', 'height']
    header_vapnik = header + ['bound']

    for i, (file, writer) in enumerate(zip(files, csv_writers)):
        if i == 1:
            writer.writerow(header_vapnik)
        else:
            writer.writerow(header)
        file.flush()

    for draw in range(n_draws):
        print(f'Running draw #{draw}')
        seed = draw*10 + 1
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y,
                                                test_size=test_split_ratio,
                                                random_state=seed)
        trees = train(X_tr, y_tr,
                      n_folds=n_folds,
                      max_n_leaves=max_n_leaves,
                      error_prior_exponent=error_prior_exponent)
        
        for i, (tree, file, csv_writer) in enumerate(zip(trees, files, csv_writers)):
            acc_tr = accuracy_score(y_tr, tree.predict(X_tr))
            acc_ts = accuracy_score(y_ts, tree.predict(X_ts))
            leaves = tree.tree.n_leaves
            height = tree.tree.height
            if i == 1:
                bound = tree.bound_value
                csv_writer.writerow([draw, seed, acc_tr, acc_ts, leaves, height, bound])
            else:
                csv_writer.writerow([draw, seed, acc_tr, acc_ts, leaves, height])
            file.flush()

    for file in files:
        file.close()


if __name__ == "__main__":
    
    exp_name = 'shawe-taylor_bound'
    
    # for dataset in load_datasets(['iris', 'wine']):
    datasets = list(load_datasets())
    for dataset in datasets[1:]:
        with Timer(f'Dataset {dataset.name}'):
            launch_experiment(dataset,
                              error_prior_exponent=12.88,
                              exp_name=exp_name)