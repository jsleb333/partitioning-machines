import sys, os
sys.path.append(os.getcwd())

from experiments.datasets.datasets import Iris
from experiments.experiment import Experiment
from experiments.models import NoPruning, OursShaweTaylorPruning, ReducedErrorPruning
from copy import deepcopy


class TestExperiment:
    def test_train_test_split_is_different_across_draws(self):
        iris = Iris()
        n_draws = 3
        assert all((iris.train_size == 150, iris.val_size == 0, iris.test_size == 0))
        exp = Experiment(dataset=iris, model=NoPruning(), n_draws=n_draws)

        sets = []
        for draw in range(n_draws):
            exp._run(draw)
            assert all((iris.train_size == 120, iris.val_size == 0, iris.test_size == 30))
            sets.extend([iris.X_train.copy(), iris.X_test.copy()])

        assert all(((sets[0] != sets[2]).any(), (sets[0] != sets[4]).any()))
        assert all(((sets[1] != sets[3]).any(), (sets[1] != sets[5]).any()))

    def test_train_val_test_split_is_different_across_draws(self):
        iris = Iris()
        n_draws = 3
        assert all((iris.train_size == 150, iris.val_size == 0, iris.test_size == 0))
        exp = Experiment(dataset=iris, model=ReducedErrorPruning(), n_draws=n_draws)

        sets = []
        for draw in range(n_draws):
            exp._run(draw)
            assert all((iris.train_size == 90, iris.val_size == 30, iris.test_size == 30))
            sets.extend([iris.X_train.copy(), iris.X_val.copy(), iris.X_test.copy()])

        assert all(((sets[0] != sets[3]).any(), (sets[0] != sets[6]).any()))
        assert all(((sets[1] != sets[4]).any(), (sets[1] != sets[7]).any()))
        assert all(((sets[2] != sets[5]).any(), (sets[2] != sets[8]).any()))

    def test_train_val_test_split_is_same_across_models(self):
        iris = Iris()
        n_draws = 1
        assert all((iris.train_size == 150, iris.val_size == 0, iris.test_size == 0))

        sets_tr = []
        sets_val = []
        sets_ts = []
        for model in [ReducedErrorPruning(), NoPruning(), ReducedErrorPruning()]:
            exp = Experiment(dataset=iris, model=model, n_draws=n_draws)
            exp._run(1)
            sets_tr.append(iris.X_train.copy())
            sets_val.append(iris.X_val.copy())
            sets_ts.append(iris.X_test.copy())

        assert (sets_tr[0] == sets_tr[2]).all()
        assert (sets_val[0] == sets_val[2]).all()
        assert all((sets_ts[0] == sets_ts[j]).all() for j in range(3))
