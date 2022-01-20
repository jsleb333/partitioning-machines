import sys, os
import numpy as np
sys.path.append(os.getcwd())

from experiments.datasets.datasets import Iris


class TestDataset:
    def test_train_val_split_is_consistent_across_shuffle(self):
        seed = 42
        iris = Iris(.1, .1, shuffle=seed)
        X_1 = iris.X_val.copy()
        iris.make_train_val_split(.1, shuffle=101)
        assert (X_1 != iris.X_val).any()
        iris.make_train_val_split(.1, shuffle=seed)
        assert (X_1 == iris.X_val).all()

    def test_sets_are_disjoint(self):
        iris = Iris(.1, .2)
        assert not np.in1d(iris.train_ind, iris.val_ind).any()
        assert not np.in1d(iris.train_ind, iris.test_ind).any()
        assert not np.in1d(iris.val_ind, iris.test_ind).any()

        iris.make_train_val_split(.25)
        assert not np.in1d(iris.train_ind, iris.val_ind).any()
        assert not np.in1d(iris.train_ind, iris.test_ind).any()
        assert not np.in1d(iris.val_ind, iris.test_ind).any()

        iris.make_train_test_split(.1)
        assert not np.in1d(iris.train_ind, iris.val_ind).any()
        assert not np.in1d(iris.train_ind, iris.test_ind).any()
        assert not np.in1d(iris.val_ind, iris.test_ind).any()
