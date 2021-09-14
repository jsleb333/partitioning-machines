from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sys, os
sys.path.append(os.getcwd())
import csv
from time import time
from datetime import datetime

from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion, shawe_taylor_bound_pruning_objective_factory, breiman_alpha_pruning_objective, modified_breiman_pruning_objective_factory
from experiments.pruning import prune_with_bound, prune_with_cv

from datasets.datasets import Dataset, camel_to_snake


class Logger:
    def __init__(self, exp_path) -> None:
        self.exp_path = exp_path

        os.makedirs(self.exp_path, exist_ok=True)
        self.is_closed = True

    def dump_exp_config(self, model_name: str, exp_config: dict) -> None:
        with open(self.exp_path + f'{model_name}_exp_config.py', 'w') as file:
            file.write(f"exp_config = { {k:str(v) for k, v in exp_config.items()} }")

    def prepare_csv_file(self, model_name: str) -> None:
        self.file = open(self.exp_path + model_name + '.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.is_closed = False

    def dump_row(self, row: dict, model_name: str) -> None:
        if self.is_closed:
            self.prepare_csv_file(model_name)
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


class Experiment:
    def __new__(cls, *args, **kwargs):
        new_exp = super().__new__(cls)
        new_exp.config = kwargs | {'datetime': datetime.now()}
        return new_exp

    def __init__(self, *,
                 dataset: Dataset,
                 test_split_ratio: float = .2,
                 n_draws: int = 25,
                 max_n_leaves: int = 40) -> None:
        self.model_name = camel_to_snake(type(self).__name__)
        self.dataset = dataset
        self.test_split_ratio = test_split_ratio
        self.n_draws = n_draws
        self.max_n_leaves = max_n_leaves

    def run(self, *args, logger: Logger = None, tracker: Tracker = None, **kwargs) -> None:

        if logger:
            logger.dump_exp_config(self.model_name, self.config)
        if tracker:
            tracker.start()

        for draw in range(self.n_draws):
            metrics = self._run(draw, *args, **kwargs)
            if logger:
                logger.dump_row(metrics, self.model_name)
            if tracker:
                tracker.display_mean_time(draw)

        if tracker:
            tracker.end(draw)
        if logger:
            logger.close()

    def _run(self, draw: int, *args, **kwargs) -> dict:
        seed = draw*10 + 1
        X_tr, X_ts, y_tr, y_ts = train_test_split(self.dataset.examples, self.dataset.labels,
                                                  test_size=self.test_split_ratio,
                                                  random_state=seed)
        tree = self._fit_tree(X_tr, y_tr, *args, **kwargs)
        t_start = time()
        self._prune_tree(*args, tree=tree, **kwargs)
        elapsed_time = time() - t_start
        acc_tr, acc_ts = self._evaluate_tree(X_tr, X_ts, y_tr, y_ts, tree=tree, **kwargs)

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

    def _fit_tree(self, X_tr, y_tr) -> None:
        tree = DecisionTreeClassifier(gini_impurity_criterion,
                                      max_n_leaves=self.max_n_leaves)

        nominal_mask = [i in self.dataset.nominal_features for i in range(self.dataset.n_features)]
        tree.fit(X_tr, y_tr, nominal_mask=nominal_mask)
        tree.bound_value = 'NA'
        return tree

    def _prune_tree(self, tree) -> None:
        raise NotImplementedError

    def _evaluate_tree(self, X_tr, X_ts, y_tr, y_ts, tree) -> tuple[float, float]:
        acc_tr = accuracy_score(y_tr, tree.predict(X_tr))
        acc_ts = accuracy_score(y_ts, tree.predict(X_ts))
        return acc_tr, acc_ts


class NoPruning(Experiment):
    def _prune_tree(self, tree) -> None:
        pass

class PruneOursShaweTaylor(Experiment):
    def __init__(self, *, error_prior_exponent: float = 13.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.error_prior_exponent = error_prior_exponent

    def _prune_tree(self, tree) -> None:
        r = 1/2**self.error_prior_exponent
        errors_logprob_prior = lambda n_err: np.log(1-r) + n_err * np.log(r)
        bound = shawe_taylor_bound_pruning_objective_factory(
            self.dataset.n_features,
            self.dataset.nominal_feat_dist,
            self.dataset.ordinal_feat_dist,
            errors_logprob_prior=errors_logprob_prior)

        tree.bound_value = prune_with_bound(tree, bound)


if __name__ == '__main__':
    from datasets.datasets import Iris, Wine
    e = PruneOursShaweTaylor(dataset=Wine, n_draws=2)
    e.run(tracker=Tracker(), logger=Logger(exp_path='./test/'))
