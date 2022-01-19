import sys, os
sys.path.append(os.getcwd())

import experiments.datasets.datasets as datasets
from experiments.datasets.datasets import Iris


class TestDataset:
    def test_train_val_split_is_consistent_across_shuffle(self):
        seed = 42
        iris = Iris(.1, .1, shuffle=seed)
        X_1 = iris.X_val.copy()
        print(X_1)
        iris.make_train_val_split(.1, shuffle=101)
        print(iris.X_val)
        assert (X_1 != iris.X_val).any()
        iris.make_train_val_split(.1, shuffle=seed)
        print(iris.X_val)
        assert (X_1 == iris.X_val).all()


