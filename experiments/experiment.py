import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
from time import time
from datetime import datetime


from graal_utils import timed, Timer
import sys, os

sys.path.append(os.getcwd())

from hypergeo import hypinv_upperbound

from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion
from partitioning_machines import shawe_taylor_bound, vapnik_bound
from partitioning_machines import breiman_alpha_pruning_objective, modified_breiman_pruning_objective_factory
from partitioning_machines import growth_function_upper_bound, wedderburn_etherington

from experiments.pruning import prune_with_cv, prune_with_score, ErrorScore, BoundScore
from experiments.datasets.datasets import Dataset
from experiments.utils import camel_to_snake


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
    def start(self, model_name):
        self.times = []
        self.draw_start = time()
        print(f'Running model {model_name.replace("_", " ")}')

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
            tracker.start(self.model_name)

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

        self._prepare_data(seed)

        self._fit_tree(*args, **kwargs)

        t_start = time()
        self._prune_tree(*args, **kwargs)
        elapsed_time = time() - t_start

        acc_tr, acc_ts = self._evaluate_tree(*args, **kwargs)

        metrics = {'draw': draw,
                   'seed': seed,
                   'train_accuracy': acc_tr,
                   'test_accuracy': acc_ts,
                   'n_leaves': self.dtc.tree.n_leaves,
                   'height': self.dtc.tree.height,
                   'bound': self.dtc.bound_value,
                   'time': elapsed_time}
        return metrics

    def _prepare_data(self, seed, *args, **kwargs) -> None:
        self.X_tr, self.X_ts, self.y_tr, self.y_ts = train_test_split(
            self.dataset.examples, self.dataset.labels,
            test_size=self.test_split_ratio,
            random_state=seed
        )

    def _fit_tree(self, *args, **kwargs) -> None:
        self.dtc = DecisionTreeClassifier(gini_impurity_criterion,
                                      max_n_leaves=self.max_n_leaves)

        nominal_mask = [i in self.dataset.nominal_features for i in range(self.dataset.n_features)]
        self.dtc.fit(self.X_tr, self.y_tr, nominal_mask=nominal_mask)
        self.dtc.bound_value = 'NA'

    def _prune_tree(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def _evaluate_tree(self, *args, **kwargs) -> tuple[float, float]:
        acc_tr = accuracy_score(self.y_tr, self.dtc.predict(self.X_tr))
        acc_ts = accuracy_score(self.y_ts, self.dtc.predict(self.X_ts))
        return acc_tr, acc_ts


class NoPruning(Experiment):
    def _prune_tree(self, *args, **kwargs) -> None:
        pass


class OursShaweTaylorPruning(Experiment):
    def __init__(self, *,
                 error_prior_exponent: float = 13.1,
                 delta: float = 0.05,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        r = 1/2**error_prior_exponent
        self.errors_logprob_prior = lambda n_err: np.log(1-r) + n_err * np.log(r)
        self.delta = delta

    def _prune_tree(self, *args, **kwargs) -> None:
        bound_score = BoundScore(
            self.dataset.n_features,
            self.dataset.nominal_feat_dist,
            self.dataset.ordinal_feat_dist,
            bound=shawe_taylor_bound,
            errors_logprob_prior=self.errors_logprob_prior,
            delta=self.delta,
        )
        self.dtc.bound_value = prune_with_score(self.dtc.tree, bound_score)


class OursHypInvPruning(Experiment):
    def __init__(self, *, delta: float = .05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.delta = delta

    def _prune_tree(self, *args, **kwargs) -> None:
        def bound_score(pruned_tree, subtree):
            growth_function = growth_function_upper_bound(
                pruned_tree,
                self.dataset.n_features,
                nominal_feat_dist=self.dataset.nominal_feat_dist,
                ordinal_feat_dist=self.dataset.ordinal_feat_dist,
                n_classes=self.dataset.n_classes,
                loose=True
            )
            n_leaves = pruned_tree.n_leaves
            complexity_prob = 1/n_leaves * 1/wedderburn_etherington(n_leaves)

            return hypinv_upperbound(
                pruned_tree.n_errors,
                pruned_tree.n_examples,
                growth_function,
                delta=self.delta * complexity_prob,
                mprime=4*pruned_tree.n_examples,
            )

        self.dtc.bound_value = prune_with_score(self.dtc.tree, bound_score)


class CARTPruning(Experiment):
    def __init__(self, *,n_folds: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_folds = n_folds

    def _prune_tree(self, *args, **kwargs) -> None:
        prune_with_cv(self.dtc, self.X_tr, self.y_tr, n_folds=self.n_folds, pruning_objective=breiman_alpha_pruning_objective)


class CARTPruningModified(CARTPruning):
        def _prune_tree(self, *args, **kwargs) -> None:
            pruning_objective = modified_breiman_pruning_objective_factory(self.dataset.n_features)
            prune_with_cv(self.dtc,
                          self.X_tr, self.y_tr,
                          n_folds=self.n_folds,
                          pruning_objective=pruning_objective)


class OraclePruning(Experiment):
    def _prune_tree(self, *args, **kwargs) -> None:
        test_error_score = ErrorScore(self.dtc, self.X_ts, self.y_ts)
        prune_with_score(self.dtc.tree, test_error_score)


class ReducedErrorPruning(Experiment):
    def __init__(self, val_split_ratio: float = .2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val_split_ratio = val_split_ratio

    def _prepare_data(self, seed, *args, **kwargs) -> None:
        super()._prepare_data(seed, *args, **kwargs)
        self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
            self.X_tr, self.y_tr,
            test_size=self.val_split_ratio,
            random_state=seed+6
        )

    def _run(self, draw: int, *args, **kwargs) -> dict:
        metrics = super()._run(draw, *args, **kwargs)
        metrics |= {'validation_accuracy': accuracy_score(self.y_val, self.dtc.predict(self.X_val))}
        return metrics

    def _prune_tree(self, *args, **kwargs) -> None:
        val_error_score = ErrorScore(self.dtc, self.X_val, self.y_val)
        prune_with_score(self.dtc.tree, val_error_score)


experiments_list = [NoPruning, OursShaweTaylorPruning, OursHypInvPruning, CARTPruning, CARTPruningModified, ReducedErrorPruning, OraclePruning]


if __name__ == '__main__':
    from datasets.datasets import Iris, Wine
    # for exp in [OursShaweTaylorPruning]:
    for exp in [OursHypInvPruning]:
        e = exp(dataset=Iris, n_draws=2)
        e.run(tracker=Tracker(), logger=Logger(exp_path='./experiments/results/test/'))
