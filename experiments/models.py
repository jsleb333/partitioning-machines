import numpy as np
from sklearn.metrics import accuracy_score
from copy import copy
import sys, os
sys.path.append(os.getcwd())

from hypergeo import hypinv_upperbound

from partitioning_machines import DecisionTreeClassifier, gini_impurity_criterion
from partitioning_machines import breiman_alpha_pruning_objective, modified_breiman_pruning_objective_factory
from partitioning_machines import growth_function_upper_bound, wedderburn_etherington

from experiments.generalization_bounds import shawe_taylor_bound, vapnik_bound
from experiments.pruning import prune_with_cv, prune_with_score, ErrorScore, BoundScore
from experiments.cross_validator import CrossValidator
from experiments.utils import camel_to_snake, geo_mean, get_default_kwargs, count_nodes_not_stump

model_dict = {}

class ModelRegister(type):
    def __init__(cls, *args, **kwargs):
        cls.model_name = camel_to_snake(cls.__name__)
        model_dict[cls.model_name] = cls


class Model(DecisionTreeClassifier, metaclass=ModelRegister):
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

    def fit_tree(self, dataset, seed=None, **kwargs) -> None:
        self.seed = seed
        self.nominal_mask = [i in dataset.nominal_features for i in range(dataset.n_features)]
        self.fit(dataset.X_train, dataset.y_train, nominal_mask=self.nominal_mask)
        self.bound_value = 'NA'

    def _prune_tree(self, dataset) -> dict:
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
        self.pfub_table = {}

    def _prune_tree(self, dataset) -> None:
        bound_score = BoundScore(
            dataset=dataset,
            bound=shawe_taylor_bound,
            table=self.pfub_table,
            errors_logprob_prior=self.errors_logprob_prior,
            delta=self.delta,
        )
        self.bound_value = prune_with_score(self, bound_score)


# class OursShaweTaylorPruningCV(OursShaweTaylorPruning):
#     def __init__(self, *,
#                  n_folds: int = 10,
#                  delta: float = 0.05,
#                  **kwargs) -> None:
#         super().__init__(delta=delta, **kwargs)
#         self.n_folds = n_folds

#     def _prune_tree(self, dataset) -> None:
#         self._cv_error_prior_exponent(dataset)
#         super()._prune_tree(dataset)

#     def _cv_error_prior_exponent(self, dataset):
#         exponents = list(range(1, 21))
#         cv_dtc = [copy(self) for _ in range(self.n_folds)]
#         fold_idx = list(KFold(n_splits=self.n_folds,
#                               shuffle=True,
#                               random_state=self.seed).split(dataset.X_train))

#         best_indices = np.zeros(len(exponents))
#         for fold, (tr_idx, ts_idx) in enumerate(fold_idx):
#             X_train, y_train = dataset.X_train[tr_idx], dataset.y_train[tr_idx]
#             X_test, y_test = dataset.X_train[ts_idx], dataset.y_train[ts_idx]
#             dtc = copy(self).fit(X_train, y_train, nominal_mask=self.nominal_mask)

#             accuracies = []
#             for exponent in exponents:
#                 r = 1/2**exponent
#                 self.errors_logprob_prior = lambda n_err: np.log(1-r) + n_err * np.log(r)
#                 copy_of_dtc = copy(dtc)
#                 copy_of_dtc.tree = copy(dtc.tree)
#                 OursShaweTaylorPruning._prune_tree(copy_of_dtc, dataset)
#                 accuracies.append(accuracy_score(y_true=y_test, y_pred=copy_of_dtc.predict(X_test)))

#             best_indices += (np.array(accuracies) == np.max(accuracies))

#         best_exponent = np.array(exponents)[best_indices == np.max(best_indices)].mean()
#         # best_exponent = (exponents[np.argmax(best_indices)] + exponents[::-1][np.argmax(best_indices[::-1])])/2

#         # print(f'{best_exponent=}')
#         r = 1/2**best_exponent
#         self.errors_logprob_prior = lambda n_err: np.log(1-r) + n_err * np.log(r)


class OursSinglePassST(Model):
    def __init__(self, *,
                 delta: float = 0.05,
                 error_priors: np.ndarray = np.logspace(-30, -1, num=15, base=2),
                 n_folds: int = 10,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.error_priors = error_priors
        self.delta = delta
        self.n_folds = n_folds
        self.pfub_table = {}

    def _prune_tree(self, dataset) -> dict:
        def pruning_objective_factory(r, dtc):
            objective = BoundScore(
                dataset=dataset,
                bound=shawe_taylor_bound,
                table=self.pfub_table,
                errors_logprob_prior=lambda n_errors: np.log(1-r) + n_errors*np.log(r),
                delta=self.delta,
            )
            bound_before_pruning = objective(dtc)
            def pruning_objective(subtree):
                pruned_dtc = copy(dtc)
                pruned_dtc.tree = subtree.remove_subtree(inplace=False).root
                return objective(pruned_dtc) - bound_before_pruning
            return pruning_objective

        cv = CrossValidator(dataset, self, self.n_folds)

        def func_to_maximize(dtc, X_test, y_test, param):
            dtc.prune_tree(0, pruning_objective_factory(param, dtc))
            return accuracy_score(y_true=y_test, y_pred=dtc.predict(X_test))

        best_rs = cv.cross_validate(func_to_maximize, self.error_priors, seed=self.seed)
        best_r = geo_mean(best_rs)
        self.prune_tree(0, pruning_objective=pruning_objective_factory(best_r, self))

        return {'error_prior_exponent': np.log2(best_r)}


class OursHypInvPruning(Model):
    def __init__(self, *,
                 delta: float = .05,
                 mprime_ratio: int = 4,
                 pfub_factor: float = 1.37e9,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.delta = delta
        self.mprime_ratio = mprime_ratio
        self.pfub_factor = pfub_factor
        self.pfub_table = {}

    def _prune_tree(self, dataset) -> None:
        def bound_score(pruned_dtc, subtree):
            log_growth_function = growth_function_upper_bound(
                pruned_dtc.tree,
                n_rl_feat=dataset.n_real_valued_features,
                nominal_feat_dist=dataset.nominal_feat_dist,
                ordinal_feat_dist=dataset.ordinal_feat_dist,
                n_classes=dataset.n_classes,
                pre_computed_tables=self.pfub_table,
                loose=True,
                log=True
            )
            complexity_prob = 1/sum(wedderburn_etherington(n) for n in range(self.max_n_leaves))
            node_dtc = count_nodes_not_stump(pruned_dtc.tree)

            return hypinv_upperbound(
                k=pruned_dtc.tree.n_errors,
                m=pruned_dtc.tree.n_examples,
                growth_function=lambda m: log_growth_function(m) - node_dtc*np.log(self.pfub_factor),
                delta=np.log(self.delta) + np.log(complexity_prob),
                mprime=self.mprime_ratio*pruned_dtc.tree.n_examples,
                log_delta=True
            )

        self.bound_value = prune_with_score(self, bound_score)


class OursSinglePassHTI(Model):
    def __init__(self, *,
                 delta: float = 0.05,
                 mprime_ratio: int = 4,
                 pfub_factors: float = np.logspace(0, 20, num=21, base=10),
                 n_folds: int = 5,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.delta = delta
        self.mprime_ratio = mprime_ratio
        self.pfub_factors = pfub_factors
        self.n_folds = n_folds
        self.pfub_table = {}

    def hti_bound(self, pfub_factor, dataset):
        log_growth_function = growth_function_upper_bound(
            self.tree,
            n_rl_feat=dataset.n_real_valued_features,
            nominal_feat_dist=dataset.nominal_feat_dist,
            ordinal_feat_dist=dataset.ordinal_feat_dist,
            n_classes=dataset.n_classes,
            pre_computed_tables=self.pfub_table,
            loose=True,
            log=True
        )
        complexity_prob = 1/sum(wedderburn_etherington(n) for n in range(self.max_n_leaves))
        node_dtc = count_nodes_not_stump(self.tree)

        return hypinv_upperbound(
            k=self.tree.n_errors,
            m=self.tree.n_examples,
            growth_function=lambda m: log_growth_function(m) - node_dtc*np.log(pfub_factor),
            delta=np.log(self.delta) + np.log(complexity_prob),
            mprime=self.mprime_ratio*self.tree.n_examples,
            log_delta=True
        )

    def _prune_tree(self, dataset) -> dict:
        def pruning_objective_factory(pfub_factor, dtc):
            bound_before_pruning = self.hti_bound(pfub_factor, dataset)
            def pruning_objective(subtree):
                pruned_dtc = copy(dtc)
                pruned_dtc.tree = subtree.remove_subtree(inplace=False).root
                return pruned_dtc.hti_bound(pfub_factor, dataset) - bound_before_pruning
            return pruning_objective

        cv = CrossValidator(dataset, self, self.n_folds)

        def func_to_maximize(dtc, X_test, y_test, param):
            dtc.prune_tree(0, pruning_objective_factory(param, dtc))
            return accuracy_score(y_true=y_test, y_pred=dtc.predict(X_test))

        best_pfub_factors = cv.cross_validate(func_to_maximize, self.pfub_factors, seed=self.seed)
        best_pfub_factor = np.mean(best_pfub_factors)
        self.prune_tree(0, pruning_objective=pruning_objective_factory(best_pfub_factor, self))

        return {'pfub_factor': np.log10(best_pfub_factor)}


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
    def __init__(self, *,
                 delta: float = .05,
                 cs: np.ndarray = np.logspace(-20, 0, num=21, base=10),
                 n_folds: int = 10,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.delta = delta
        self.cs = cs
        self.n_folds = n_folds
        self.table = {}

    def alpha(self, subtree, dataset, c):
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

        log_gf_tree_path = growth_function_upper_bound(
            tree_path,
            n_rl_feat=dataset.n_real_valued_features,
            nominal_feat_dist=dataset.nominal_feat_dist,
            ordinal_feat_dist=dataset.ordinal_feat_dist,
            n_classes=dataset.n_classes,
            loose=True,
            pre_computed_tables=self.table,
            log=True
        )(subtree.n_examples)

        log_gf_subtree = growth_function_upper_bound(
            subtree,
            n_rl_feat=dataset.n_real_valued_features,
            nominal_feat_dist=dataset.nominal_feat_dist,
            ordinal_feat_dist=dataset.ordinal_feat_dist,
            n_classes=dataset.n_classes,
            loose=True,
            pre_computed_tables=self.table,
            log=True
        )(subtree.n_examples)

        return c*np.sqrt(
            (log_gf_tree_path + log_gf_subtree + np.log(dataset.n_examples/self.delta))
            /subtree.n_examples
        )

    def _prune_tree(self, dataset) -> None:
        def pruning_objective_factory(c):
            def pruning_objective(subtree):
                """
                Equation (1) of the paper of Kearns and Mansour (1998).
                """
                frac_errors_subtree = subtree.n_errors/subtree.n_examples
                frac_errors_leaf = 1 - np.max(subtree.n_examples_by_label)/subtree.n_examples

                return frac_errors_leaf - frac_errors_subtree - self.alpha(subtree, dataset, c)
            return pruning_objective

        cv = CrossValidator(dataset, self, self.n_folds)

        def func_to_maximize(dtc, X_test, y_test, param):
            dtc.prune_tree(0, pruning_objective_factory(param))
            return accuracy_score(y_true=y_test, y_pred=dtc.predict(X_test))

        best_cs = cv.cross_validate(func_to_maximize, self.cs, seed=self.seed)

        self.prune_tree(0, pruning_objective=pruning_objective_factory(np.mean(best_cs)))


class ReducedErrorPruning(Model):
    def __init__(self, *,
                 val_split_ratio: float = .2,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.val_split_ratio = val_split_ratio

    def fit_tree(self, dataset, seed=42, **kwargs) -> None:
        dataset.make_train_val_split(self.val_split_ratio, seed)
        super().fit_tree(dataset)

    def _prune_tree(self, dataset) -> None:
        val_error_score = ErrorScore(dataset.X_val, dataset.y_val)
        prune_with_score(self, val_error_score)


class OraclePruning(Model):
    def _prune_tree(self, dataset) -> None:
        test_error_score = ErrorScore(dataset.X_test, dataset.y_test)
        prune_with_score(self, test_error_score)


if __name__ == '__main__':
    from experiments.datasets import Wine, Iris, Amphibians

    for d in [Wine, Iris, Amphibians]:
        print(d)
        dataset = d(0, .5, 37)
        # for n in [2, 5, 8, 10]:
        #     print(f'{n=}')
        #     model = OursShaweTaylorPruningCV(n_folds=n)
        #     model.fit_tree(dataset, seed=101)
        #     print('leaves =', model.tree.n_leaves)
        #     model._prune_tree(dataset)
        #     print('leaves =', model.tree.n_leaves)
        #     print(model.evaluate_tree(dataset))
        seed = 42
        model = OursHypInvPruning()
        model.fit_tree(dataset, seed=seed)
        model._prune_tree(dataset)
        print('HTI', model.evaluate_tree(dataset))

        model = OursSinglePassHTI()
        model.fit_tree(dataset, seed=seed)
        print(model._prune_tree(dataset))
        print('SP-HTI', model.evaluate_tree(dataset))

        # model = ReducedErrorPruning()
        # dataset.make_train_val_split(.15)
        # model.fit_tree(dataset, seed=seed)
        # model._prune_tree(dataset)
        # print('REDER', model.evaluate_tree(dataset))


