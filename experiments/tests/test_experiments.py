import sys, os
sys.path.append(os.getcwd())

import experiments.datasets.datasets as dataset
from experiments.experiment import *
from copy import deepcopy


class TestExperiment:
    def test_train_test_split_is_different_across_runs(self):
        exp = Experiment(dataset=dataset.Iris, n_draws=3)
        run = []
        for draw in range(exp.n_draws):
            exp._prepare_data(draw*10 + 1)
            run.extend([deepcopy(exp.X_tr),
                        deepcopy(exp.X_ts),
                        deepcopy(exp.y_tr),
                        deepcopy(exp.y_ts)])

        for i in range(4):
            assert (run[i] != run[i+4]).any()
            assert (run[i] != run[i+8]).any()

    def test_train_test_split_is_consistent_across_runs(self):
        exp = Experiment(dataset=dataset.Iris, n_draws=3)
        first_run, second_run = [], []
        for run in (first_run, second_run):
            for draw in range(exp.n_draws):
                exp._prepare_data(draw*10 + 1)
                run.extend([deepcopy(exp.X_tr),
                            deepcopy(exp.X_ts),
                            deepcopy(exp.y_tr),
                            deepcopy(exp.y_ts)])

        assert all((f == s).any() for f, s in zip(first_run, second_run))
