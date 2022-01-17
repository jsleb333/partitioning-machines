import numpy as np
from sklearn.metrics import accuracy_score
from copy import copy

from hypergeo import hypinv_upperbound

import sys, os
sys.path.append(os.getcwd())

from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion
from partitioning_machines import shawe_taylor_bound, vapnik_bound
from partitioning_machines import breiman_alpha_pruning_objective, modified_breiman_pruning_objective_factory
from partitioning_machines import growth_function_upper_bound, wedderburn_etherington

from experiments.pruning import prune_with_cv, prune_with_score, ErrorScore, BoundScore
from experiments.utils import camel_to_snake, get_default_kwargs


model_dict = {}

class CamelToSnake(type):
    def __init__(cls, *args, **kwargs):
        cls.model_name = camel_to_snake(cls.__name__)
        model_dict[cls.model_name] = cls


class Model(DecisionTreeClassifier, metaclass=CamelToSnake):
    def __new__(cls, *args, **kwargs):
        new_model = super().__new__(cls)
        new_model.config = get_default_kwargs(cls) | kwargs
        return new_model

    def __init__(self, *,
                 model_name: str = None,
                 max_n_leaves: int = 40,
                 impurity_criterion=gini_impurity_criterion,
                 **kwargs
                 ) -> None:
        super().__init__(
            max_n_leaves=max_n_leaves,
            impurity_criterion=impurity_criterion,
            **kwargs)
        if model_name is None:
            self.model_name = camel_to_snake(type(self).__name__)
            self.config['model_name'] = self.model_name
        else:
            self.model_name = model_name

    def __str__(self):
        return self.model_name

    def __repr__(self) -> str:
        return type(self).__name__ + '()'

    def fit_tree(self, dataset) -> None:
        nominal_mask = [i in dataset.nominal_features for i in range(dataset.n_features)]
        self.fit(dataset.X_train, dataset.y_train, nominal_mask=nominal_mask)
        self.bound_value = 'NA'

    def _prune_tree(self, dataset) -> None:
        raise NotImplementedError

    def evaluate_tree(self, dataset) -> tuple[float, float]:
        acc_tr = accuracy_score(dataset.y_train, self.predict(dataset.X_train))
        acc_val = None
        if dataset.val_size > 0:
            acc_val = accuracy_score(dataset.y_val, self.predict(dataset.X_val))
        acc_ts = None
        if dataset.test_size > 0:
            acc_ts = accuracy_score(dataset.y_test, self.predict(dataset.X_test))
        return acc_tr, acc_val, acc_ts

del model_dict['model']

class NoPruning(Model):
    def _prune_tree(self, dataset) -> None:
        pass


class OursShaweTaylorPruning(Model):
    def __init__(self, *,
                 error_prior_exponent: float = 13.1,
                 delta: float = 0.05,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        r = 1/2**error_prior_exponent
        self.errors_logprob_prior = lambda n_err: np.log(1-r) + n_err * np.log(r)
        self.delta = delta

    def _prune_tree(self, dataset) -> None:
        bound_score = BoundScore(
            dataset.n_features,
            dataset.nominal_feat_dist,
            dataset.ordinal_feat_dist,
            bound=shawe_taylor_bound,
            errors_logprob_prior=self.errors_logprob_prior,
            delta=self.delta,
        )
        self.bound_value = prune_with_score(self.tree, bound_score)


class OursHypInvPruning(Model):
    def __init__(self, *, delta: float = .05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.delta = delta

    def _prune_tree(self, dataset) -> None:
        def bound_score(pruned_tree, subtree):
            growth_function = growth_function_upper_bound(
                pruned_tree,
                dataset.n_features,
                nominal_feat_dist=dataset.nominal_feat_dist,
                ordinal_feat_dist=dataset.ordinal_feat_dist,
                n_classes=dataset.n_classes,
                loose=True
            )
            # complexity_prob = 1/self.max_n_leaves * 1/wedderburn_etherington(pruned_tree.n_leaves)
            complexity_prob = 1/sum(wedderburn_etherington(n) for n in range(self.max_n_leaves))

            return hypinv_upperbound(
                pruned_tree.n_errors,
                pruned_tree.n_examples,
                growth_function,
                delta=self.delta * complexity_prob,
                mprime=4*pruned_tree.n_examples,
            )

        self.bound_value = prune_with_score(self.tree, bound_score)


class CARTPruning(Model):
    def __init__(self, *,n_folds: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_folds = n_folds

    def _prune_tree(self, dataset) -> None:
        prune_with_cv(self, dataset.X_train, dataset.y_train, n_folds=self.n_folds, pruning_objective=breiman_alpha_pruning_objective)


class CARTPruningModified(CARTPruning):
    def _prune_tree(self, dataset) -> None:
        pruning_objective = modified_breiman_pruning_objective_factory(dataset.n_features)
        prune_with_cv(self,
                      dataset.X_train, dataset.y_train,
                      n_folds=self.n_folds,
                      pruning_objective=pruning_objective)


class KearnsMansourPruning(Model):
    def __init__(self, *, delta: float = .05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.delta = delta

    def alpha(self, subtree, dataset):
        """
        Equation (2) of the paper of Kearns and Mansour (1998) using our growth function upper bound.
        """
        tree_path = copy(subtree.root)
        for direction in subtree.path_from_root():
            if direction == 'left':
                tree_path.right_subtree.remove_subtree()
                tree_path = tree_path.left_subtree
            else:
                tree_path.left_subtree.remove_subtree()
                tree_path = tree_path.right_subtree
            tree_path.remove_subtree()
            tree_path = tree_path.root

        gf_tree_path = growth_function_upper_bound(
            tree_path,
            dataset.n_features,
            nominal_feat_dist=dataset.nominal_feat_dist,
            ordinal_feat_dist=dataset.ordinal_feat_dist,
            n_classes=dataset.n_classes,
            loose=True
        )(subtree.n_examples)

        gf_subtree = growth_function_upper_bound(
            subtree,
            dataset.n_features,
            nominal_feat_dist=dataset.nominal_feat_dist,
            ordinal_feat_dist=dataset.ordinal_feat_dist,
            n_classes=dataset.n_classes,
            loose=True
        )(subtree.n_examples)

        return np.sqrt(
            (np.log(float(gf_tree_path))
             + np.log(float(gf_subtree))
             + np.log(dataset.n_examples/self.delta)
            )/subtree.n_examples)

    def _prune_tree(self, dataset) -> None:
        """
        Equation (1) of the paper of Kearns and Mansour (1998).
        """
        for subtree in self.tree.traverse(order='post'):
            if subtree.is_leaf():
                continue
            frac_errors_subtree = subtree.n_errors/subtree.n_examples
            frac_errors_leaf = 1 - np.max(subtree.n_examples_by_label)/subtree.n_examples

            subtree.pruning_coef = frac_errors_leaf - frac_errors_subtree - self.alpha(subtree, dataset)

        self._prune_tree(0)

class OraclePruning(Model):
    def _prune_tree(self, dataset) -> None:
        test_error_score = ErrorScore(self, dataset.X_test, dataset.y_test)
        prune_with_score(self.tree, test_error_score)


class ReducedErrorPruning(Model):
    def _prune_tree(self, dataset) -> None:
        val_error_score = ErrorScore(self, dataset.X_val, dataset.y_val)
        prune_with_score(self.tree, val_error_score)
