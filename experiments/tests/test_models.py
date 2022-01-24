import sys, os
sys.path.append(os.getcwd())
from sklearn.metrics import accuracy_score
from copy import copy

from experiments.datasets.datasets import Iris, QSARBiodegradation
from experiments.models import ReducedErrorPruning, NoPruning


class TestModel:
    def test_reduced_error_pruning(self):
        iris = QSARBiodegradation(.2, .2)
        reder = ReducedErrorPruning()
        reder.fit_tree(iris)

        nopru = NoPruning()
        nopru.fit_tree(iris)

        assert reder.tree == nopru.tree

        reder._prune_tree(iris)

        assert reder.tree != nopru.tree

        current_val_acc = accuracy_score(y_true=iris.y_val, y_pred=nopru.predict(iris.X_val))
        best_val_acc = current_val_acc

        copy_of_nopru = copy(nopru)
        best_tree = nopru.tree

        finished_pruning = False
        i = 0
        while not finished_pruning:
            i += 1
            if i > 100:
                break
            finished_pruning = True
            current_tree = best_tree
            for node in current_tree:
                if node.is_leaf():
                    continue
                copy_of_tree = copy(current_tree) # Create a copy of tree
                copy_of_tree.follow_path(node.path_from_root()).remove_subtree() # Prune subtree
                assert copy_of_tree.n_leaves < current_tree.n_leaves

                copy_of_nopru.tree = copy_of_tree
                current_val_acc = accuracy_score(y_true=iris.y_val, y_pred=copy_of_nopru.predict(iris.X_val)) # Evaluate performance of pruned tree

                if current_val_acc >= best_val_acc:
                    best_tree = copy_of_tree
                    best_val_acc = current_val_acc
                    finished_pruning = False

        nopru.tree = best_tree

        for acc_1, acc_2 in zip(nopru.evaluate_tree(iris), reder.evaluate_tree(iris)):
            assert acc_1 == acc_2

