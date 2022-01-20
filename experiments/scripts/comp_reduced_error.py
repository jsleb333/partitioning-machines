import sys, os

sys.path.append(os.getcwd())
from graal_utils import Timer
import numpy as np

from experiments.models import NoPruning, ReducedErrorPruning, OursShaweTaylorPruning, OraclePruning
from experiments.experiment import Experiment, Tracker, Logger
from experiments.datasets.datasets import QSARBiodegradation


class NoPruningVal(NoPruning):
    def __init__(self, *, val_split_ratio=.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val_split_ratio = val_split_ratio

    def fit_tree(self, dataset) -> None:
        dataset.make_train_val_split()
        super().fit_tree(dataset)


class STPruningVal(OursShaweTaylorPruning):
    def __init__(self, *, val_split_ratio=.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val_split_ratio = val_split_ratio

    def fit_tree(self, dataset) -> None:
        dataset.make_train_val_split()
        super().fit_tree(dataset)


class OraclePruningVal(OraclePruning):
    def __init__(self, *, val_split_ratio=.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val_split_ratio = val_split_ratio

    def fit_tree(self, dataset) -> None:
        dataset.make_train_val_split()
        super().fit_tree(dataset)


if __name__ == "__main__":

    dataset = QSARBiodegradation
    exp_path = f'./experiments/results/reder-exp01/{dataset.name}/'
    n_draws = 10
    test_split_ratio = .2
    max_n_leaves = [10, 20, 30, 40, 50, 60]
    val_split_ratios = np.linspace(0,1,11)
    models = [NoPruningVal, STPruningVal, ReducedErrorPruning, OraclePruningVal]

    # exp_path = f'./experiments/results/test/'
    # n_draws = 1
    # max_n_leaves = [50, 55, 60]
    # val_split_ratios = [.2]
    # models = [STPruningVal]

    for model in models:
        for n_leaves in max_n_leaves:
            for val_split_ratio in val_split_ratios:
                if val_split_ratio == 0 and model is ReducedErrorPruning:
                    continue

                exp_name = f'{model.model_name}-val={val_split_ratio:.2f}-n_leaves={n_leaves}'
                with Timer(exp_name):
                    Experiment(
                        dataset=dataset,
                        model=model(val_split_ratio=val_split_ratio, max_n_leaves=n_leaves),
                        test_split_ratio=test_split_ratio,
                        n_draws=n_draws,
                        exp_name=exp_name
                    ).run(logger=Logger(exp_path),
                          tracker=Tracker())
