"""
This file contains the code necessary to run all experiments of the paper 'Decision trees as partitioning machines to characterize their generalization properties' by Leboeuf, LeBlanc and Marchand (2020). See the README for usage details.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
import csv
from time import time

from graal_utils import Timer

from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion, shawe_taylor_bound_pruning_objective_factory, breiman_alpha_pruning_objective, modified_breiman_pruning_objective_factory
from experiments.pruning import prune_with_bound, prune_with_cv
from partitioning_machines import func_to_cmd

from datasets.datasets import load_datasets, Dataset


class Experiment:
    def __new__(cls, *args, **kwargs):
        new_exp = super().__new__(cls, *args, **kwargs)
        new_exp.config = kwargs
        return new_exp

    def __init__(self, *,
                 model_name: str,
                 exp_name: str,
                 dataset: Dataset,
                 test_split_ratio: float = .2,
                 n_draws: int = 25,
                 max_n_leaves: int = 40,
                 exp_path: str = '') -> None:
        self.model_name = model_name
        self.exp_name = exp_name
        self.dataset = dataset
        self.test_split_ratio = test_split_ratio
        self.n_draws = n_draws
        self.max_n_leaves = max_n_leaves

        self.exp_path = exp_path or f'./experiments/results/{dataset.name}/{exp_name}/'

    def _fit_tree(self, X_tr, y_tr, seed) -> None:
        tree = DecisionTreeClassifier(gini_impurity_criterion,
                                      max_n_leaves=self.max_n_leaves)

        nominal_mask = [1 if i in self.dataset.nominal_features else 0
                        for i in range(self.dataset.n_features)]
        tree.fit(X_tr, y_tr, nominal_mask=nominal_mask)
        tree.bound_value = 'NA'
        return tree

    def _prune_tree(self, tree) -> None:
        raise NotImplementedError

    def _evaluate_tree(self, X_tr, X_ts, y_tr, y_ts, tree):
        acc_tr = accuracy_score(y_tr, tree.predict(X_tr))
        acc_ts = accuracy_score(y_ts, tree.predict(X_ts))
        return acc_tr, acc_ts

    def _run(self, draw: int, *args, **kwargs) -> dict:
        seed = draw*10 + 1
        X_tr, X_ts, y_tr, y_ts = train_test_split(self.dataset.examples, self.dataset.labels,
                                                  test_size=self.test_split_ratio,
                                                  random_state=seed)
        tree = self._fit_tree(X_tr, y_tr, *args, seed=seed, **kwargs)
        t_start = time()
        self._prune_tree(*args, tree=tree, **kwargs)
        elapsed_time = time() - t_start
        acc_tr, acc_ts = self._evaluate_tree(X_ts, y_ts, *args, tree=tree, **kwargs)

        leaves = tree.tree.n_leaves
        height = tree.tree.height
        bound = tree.bound_value

        metrics = {'draw': draw,
                   'seed': seed,
                   'train_accuracy': acc_tr,
                   'test_accuracy': acc_ts,
                   'n_leaves': leaves,
                   'height': height,
                   'bound': bound,
                   'time': elapsed_time}
        return metrics

    def run(self, *args, logger=None, tracker=None, **kwargs) -> None:

        if logger:
            logger.dump_exp_config(self)

        if tracker:
            tracker.start()

        for draw in range(self.n_draws):
            metrics = self._run(draw, *args, **kwargs)

            if logger:
                logger.dump_row(metrics, self)

            if tracker:
                tracker.display_mean_time(draw)

        if tracker:
            tracker.end()

        if logger:
            logger.close()


class Logger:
    def __init__(self, exp_path) -> None:
        self.exp_path = exp_path
        os.makedirs(self.exp_path, exist_ok=True)
        self.is_closed = True

    def dump_exp_config(self, exp: Experiment) -> None:
        with open(self.exp_path + f'{self.model_name}_exp_config.py', 'w') as file:
            file.write(f"exp_config = {self.config}")

    def prepare_csv_file(self, exp: Experiment) -> None:
        self.file = open(self.exp_path + exp.model_name + '.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.is_closed = False

    def dump_row(self, row: dict, exp: Experiment) -> None:
        if self.is_closed:
            self.prepare_csv_file(exp)
            self._dump_row(row.keys())
        self._dump_row(row.values())

    def _dump_row(self, row: list) -> None:
        self.csv_writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
        self.is_closed = True


class Tracker:
    def start(self):
        self.times = []
        self.draw_start = time()

    def display_mean_time(self, draw: int) -> None:
        self.times.append(time() - self.draw_start)
        time_str = f'\tMean time per draw: {sum(self.times)/len(self.times):.3f}s.' if self.times else ''
        print(f'Running draw #{draw:02d}...' + time_str, end='\r')

        self.draw_start = time()

    def end(self, draw: int) -> None:
        print(f'\rCompleted all {draw+1} draws.')




def launch_single_experiment(dataset,
                             model_name,
                             exp_name,
                             test_split_ratio=.25,
                             n_draws=25,
                             n_folds=10,
                             max_n_leaves=40,
                             error_prior_exponent=13.1,
                             ):
    exp_params = {
        'exp_name':exp_name,
        'test_split_ratio':test_split_ratio,
        'n_draws':n_draws,
        'n_folds':n_folds,
        'max_n_leaves':max_n_leaves,
        'error_prior_exponent':error_prior_exponent,
        }

    X, y = dataset.examples, dataset.labels
    nominal_mask = [1 if i in dataset.nominal_features else 0 for i in range(dataset.n_features)]

    exp_path = f'./experiments/results/{dataset.name}/{exp_name}/'

    os.makedirs(exp_path, exist_ok=True)

    with open(exp_path + f'{model_name}_exp_params.py', 'w') as file:
        file.write(f"exp_params = {exp_params}")

    file = open(exp_path + model_name + '.csv', 'w', newline='')
    csv_writer = csv.writer(file)

    header = ['draw', 'seed', 'train_accuracy', 'test_accuracy', 'n_leaves', 'height', 'bound', 'time']

    csv_writer.writerow(header)
    file.flush()

    times_per_draw = []

    for draw in range(n_draws):
        time_str = f'\tMean time per draw: {sum(times_per_draw)/len(times_per_draw):.3f}s.' if times_per_draw else ''
        print(f'Running draw #{draw:02d}...' + time_str, end='\r')
        draw_start = time()

        seed = draw*10 + 1
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y,
                                                test_size=test_split_ratio,
                                                random_state=seed)

        decision_tree = DecisionTreeClassifier(gini_impurity_criterion, max_n_leaves=max_n_leaves)
        n_examples, n_features = X.shape

        decision_tree.fit(X_tr, y_tr, nominal_mask=nominal_mask)
        decision_tree.bound_value = 'NA'
        t_start = time()

        if model_name == 'ours':
            r = 1/2**error_prior_exponent
            errors_logprob_prior = lambda n_err: np.log(1-r) + n_err * np.log(r)
            bound = shawe_taylor_bound_pruning_objective_factory(n_features, errors_logprob_prior=errors_logprob_prior)

            decision_tree.bound_value = prune_with_bound(decision_tree, bound)

        elif model_name == 'cart':
            prune_with_cv(decision_tree, X_tr, y_tr, n_folds=n_folds, pruning_objective=breiman_alpha_pruning_objective)

        elif model_name == 'm-cart':
            modified_breiman_pruning_objective = modified_breiman_pruning_objective_factory(n_features)
            prune_with_cv(decision_tree, X_tr, y_tr, n_folds=n_folds,     pruning_objective=modified_breiman_pruning_objective)
        elif model_name != 'original':
            raise ValueError(f"Unknown model {model_name}. Valid models are 'original', 'cart', 'm-cart' and 'ours'. ")

        elapsed_time = time() - t_start

        acc_tr = accuracy_score(y_tr, decision_tree.predict(X_tr))
        acc_ts = accuracy_score(y_ts, decision_tree.predict(X_ts))
        leaves = decision_tree.tree.n_leaves
        height = decision_tree.tree.height
        bound = decision_tree.bound_value
        csv_writer.writerow([draw, seed, acc_tr, acc_ts, leaves, height, bound, elapsed_time])
        file.flush()

        times_per_draw.append(time() - draw_start)

    print(f'\rCompleted all {draw+1} draws.')
    file.close()


@func_to_cmd
def launch_experiment(dataset=list(),
                      model_names='all',
                      exp_name='',
                      test_split_ratio=.25,
                      n_draws=25,
                      n_folds=10,
                      max_n_leaves=40,
                      error_prior_exponent=13.1,
                      ):
    """
    Will launch the experiment with specified parameters. Automatically saves all results in the file: "./experiments/results/<dataset_name>/<exp_name>/<model_name>.csv". Experiments parameters are saved in the file: "./experiments/results/<dataset_name>/<exp_name>/<model_name>_params.py". See the README for more usage details.

    Args:
        dataset (list[str]): The dataset name to be used in the experiment. By default (an empty list) will iterate over all available datasets. Otherwise, will launch experiment for the specified datasets. To see all available datasets, consult the file "./experiments/datasets/datasets.py".

        model_names (str): Valid model names are 'original', 'cart', 'm-cart' and 'ours'. By default 'all' will run all 4 models one after the other.

        exp_name (str): Name of the experiment. Will be used to save the results on disk. If empty, the date and time of the beginning of the experiment will be used.

        test_split_ratio (float): Ratio of examples that will be kept for test.

        n_draws (int): Number of times the experiments will be run with a new random state.

        n_folds (int): Number of folds used by the pruning algorithms of CART and M-CART. (Ignored by 'ours' algorithm).

        max_n_leaves (int): Maximum number of leaves the original tree is allowed to have.

        error_prior_exponent (int): The distribution q_k will be of the form (1-r) * r**k, where r = 2**(-error_prior_exponent). (Ignored by 'cart' and 'm-cart' algorithms).
    """
    if not dataset:
        datasets = list(load_datasets())
    else:
        datasets = list(load_datasets(dataset))

    if model_names == 'all':
        model_names = ['original', 'ours', 'cart', 'm-cart']
    else:
        model_names = [model_names]

    if not exp_name:
        exp_name = exp_name if exp_name else datetime.now().strftime("%Y-%m-%d_%Hh%Mm")

    for model_name in model_names:
        for dataset in datasets:
            with Timer(f'{model_name} model on dataset {dataset.name}'):
                launch_single_experiment(dataset=dataset,
                                         model_name=model_name,
                                         exp_name=exp_name,
                                         test_split_ratio=test_split_ratio,
                                         n_draws=n_draws,
                                         n_folds=n_folds,
                                         max_n_leaves=max_n_leaves,
                                         error_prior_exponent=error_prior_exponent,
                                         )

if __name__ == "__main__":
    # launch_experiment()
    e = Experiment('test', load_datasets(['iris']))

