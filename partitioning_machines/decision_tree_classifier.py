import numpy as np

from partitioning_machines import Tree
from partitioning_machines import OneHotEncoder


class _DecisionTree(Tree):
    def __init__(self,
                 impurity_score,
                 n_examples_by_label,
                 rule_threshold=None,
                 rule_feature=None,
                 left_subtree=None,
                 right_subtree=None,
                 parent=None):
        super().__init__(left_subtree=left_subtree,
                         right_subtree=right_subtree,
                         parent=parent)

        self.impurity_score = impurity_score
        self.n_examples_by_label = n_examples_by_label
        self.label = np.zeros(n_examples_by_label.shape[0], dtype=int)
        self.label[np.argmax(n_examples_by_label)] = 1
        self.rule_threshold = rule_threshold
        self.rule_feature = rule_feature

    @property
    def n_examples(self):
        return self.n_examples_by_label.sum()

    @property
    def n_errors(self):
        """
        Returns the number of errors made by the subtree on the training dataset.
        """
        if self.is_leaf():
            return self.n_examples - np.max(self.n_examples_by_label)
        else:
            return self.left_subtree.n_errors + self.right_subtree.n_errors

    def predict(self, x):
        if self.is_leaf():
            return self.label
        else:
            if x[self.rule_feature] < self.rule_threshold:
                return self.left_subtree.predict(x)
            else:
                return self.right_subtree.predict(x)

    def predict_proba(self, x):
        if self.is_leaf():
            return self.n_examples_by_label / self.n_examples
        else:
            if x[self.rule_feature] < self.rule_threshold:
                return self.left_subtree.predict_proba(x)
            else:
                return self.right_subtree.predict_proba(x)

    def is_pure(self):
        return (self.n_examples_by_label == self.n_examples).any()


class DecisionTreeClassifier:
    def __init__(self,
                 impurity_criterion=None,
                 optimization_mode='min',
                 max_n_leaves=None,
                 max_depth=None,
                 min_examples_per_leaf=1):
        self.impurity_criterion = impurity_criterion
        self.optimization_mode = optimization_mode
        self.max_n_leaves = max_n_leaves if max_n_leaves is not None else np.infty
        self.max_depth = max_depth if max_depth is not None else np.infty
        self.min_examples_per_leaf = min_examples_per_leaf
        self.tree = None

    def fit(self, X, y, X_idx_sorted=None):
        n_examples, n_features = X.shape
        self.label_encoder = OneHotEncoder(y)
        encoded_y, _ = self.label_encoder.encode_labels(y)

        if X_idx_sorted is None:
            X_idx_sorted = np.argsort(X, 0)

        self._init_tree(encoded_y, n_examples)

        splitter = Splitter(X, encoded_y, self.impurity_criterion, self.optimization_mode, self.min_examples_per_leaf)

        possible_splits = [splitter.split(self.tree, X_idx_sorted)] # List of splits that can be produced.
        while possible_splits and self.tree.n_leaves < self.max_n_leaves:
            best_split = possible_splits[0]
            for split in possible_splits:
                if self.optimization_mode == 'min':
                    if best_split.impurity_score > split.impurity_score:
                        best_split = split
                elif self.optimization_mode == 'max':
                    if best_split.impurity_score < split.impurity_score:
                        best_split = split

            if best_split.split_makes_gain():
                best_split.apply_split()

            if self.tree.n_leaves < self.max_n_leaves and self.tree.height < self.max_depth:
                    X_idx_sorted_left, X_idx_sorted_right = best_split.compute_split_X_idx_sorted()

                    left_split = splitter.split(best_split.leaf.left_subtree, X_idx_sorted_left)
                    if left_split:
                        possible_splits.append(left_split)

                    right_split = splitter.split(best_split.leaf.right_subtree, X_idx_sorted_right)
                    if right_split:
                        possible_splits.append(right_split)

            possible_splits.remove(best_split)

        return self

    def _init_tree(self, encoded_y, n_examples):
        n_examples_by_label = np.sum(encoded_y, axis=0)
        self.tree = _DecisionTree(self.impurity_criterion(n_examples_by_label/n_examples),
                         n_examples_by_label)

    def predict(self, X):
        encoded_prediction = np.array([self.tree.predict(x) for x in X])
        return self.label_encoder.decode_labels(encoded_prediction)

    def predict_proba(self, X):
        return np.array([self.tree.predict_proba(x) for x in X])

    def compute_pruning_coefficients(self, prune_objective):
        """
        Computes and assigns a pruning coefficient to every internal node of the tree. The sorted list of coefficients is returned.

        Args:
            prune_objective (callable): Receives a Tree object and outputs a coefficient based on the performance of the tree. Will not be called with a leaf.

        Returns the list of pruning coefficients in increasing order.
        """
        pruning_coefs = []
        for subtree in self.tree:
            if not subtree.is_leaf():
                subtree.pruning_coef = prune_objective(subtree)
                pruning_coefs.append(subtree.pruning_coef)
        pruning_coefs.sort()

        return pruning_coefs

    def prune_tree(self, pruning_coef_threshold, pruning_objective=None):
        """
        Prunes the tree by replacing each subtree that have a pruning coefficient less than or equal to 'pruning_coef_threshold' by a leaf. Returns the number of internal nodes pruned.

        Args:
            pruning_coef_threshold (float): Threshold the pruning coefficient must satisfy.
            pruning_objective (callable): Will be used to compute the pruning coefficients if provided. Used by the 'compute_pruning_coefficients' method. If None, it assumes the 'compute_pruning_coefficients' method has already been called and subtrees possesses the 'pruning_coef' attributes.

        Returns: (int) the number of internal nodes pruned.
        """
        if pruning_objective is not None:
            self.compute_pruning_coefficients(pruning_objective)

        subtrees_to_remove = []
        n_nodes_before = self.tree.n_nodes

        for subtree in self.tree:
            if not subtree.is_leaf():
                if subtree.pruning_coef <= pruning_coef_threshold:
                    self._prune_subtree(subtree)

        return n_nodes_before - self.tree.n_nodes

    def _prune_subtree(self, subtree):
        subtree.left_subtree = None
        subtree.right_subtree = None
        self.tree.update_tree()


class Splitter:
    def __init__(self, X, y, impurity_criterion, optimization_mode, min_examples_per_leaf=1):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.impurity_criterion = impurity_criterion
        self.optimization_mode = optimization_mode
        self.min_examples_per_leaf = min_examples_per_leaf

    def split(self, leaf, X_idx_sorted):
        return Split(leaf, X_idx_sorted, self)

class Split:
    def __init__(self, leaf, X_idx_sorted, splitter):
        self.leaf = leaf
        self.X_idx_sorted = X_idx_sorted
        self.splitter = splitter
        self.n_examples, self.n_features = X_idx_sorted.shape

        self.X = splitter.X
        self.y = splitter.y
        self.n_classes = self.y.shape[1]
        self.impurity_criterion = splitter.impurity_criterion
        self.optimization_mode = splitter.optimization_mode
        self.min_examples_per_leaf = splitter.min_examples_per_leaf

        self.validity = self._find_best_split()

    def _find_best_split(self):
        n_examples_by_label = self.leaf.n_examples_by_label

        if self.leaf.is_pure():
            return False

        if self.n_examples < 2*self.min_examples_per_leaf:
            return False

        n_examples_left = self.min_examples_per_leaf - 1
        n_examples_right = self.n_examples - n_examples_left

        n_examples_by_label_left = np.zeros((self.n_features, self.n_classes))
        for i in range(n_examples_left):
            n_examples_by_label_left += self.y[self.X_idx_sorted[i]]
        n_examples_by_label_right = n_examples_by_label - n_examples_by_label_left

        self.impurity_score = np.infty if self.optimization_mode == 'min' else -np.infty

        a_rule_has_been_found = False

        for x_idx in self.X_idx_sorted[n_examples_left:-self.min_examples_per_leaf]:
            n_examples_left += 1
            n_examples_right -= 1
            transfered_labels = self.y[x_idx]
            n_examples_by_label_left += transfered_labels
            n_examples_by_label_right -= transfered_labels
            tmp_feature, tmp_impurity_score = self.argext(self._split_impurity_criterion(n_examples_by_label_left, n_examples_by_label_right, n_examples_left, n_examples_right))
            if (self.optimization_mode == 'min' and tmp_impurity_score < self.impurity_score) \
                or (self.optimization_mode == 'max' and tmp_impurity_score > self.impurity_score):
                    rule_threshold_idx_left = x_idx[tmp_feature]
                    rule_threshold_idx_right = self.X_idx_sorted[n_examples_left, tmp_feature]
                    if not np.isclose(self.X[rule_threshold_idx_left, tmp_feature],
                                      self.X[rule_threshold_idx_right, tmp_feature]):
                        self.rule_feature = tmp_feature
                        self.rule_threshold = (self.X[rule_threshold_idx_left, tmp_feature] +
                                               self.X[rule_threshold_idx_right, tmp_feature])/2
                        self.impurity_score = tmp_impurity_score
                        self.n_examples_by_label_left = n_examples_by_label_left[tmp_feature].copy()
                        self.n_examples_by_label_right = n_examples_by_label_right[tmp_feature].copy()
                        a_rule_has_been_found = True

        return a_rule_has_been_found

    def __bool__(self):
        return self.validity

    @property
    def n_examples_left(self):
        return self.n_examples_by_label_left.sum(dtype=int)

    @property
    def n_examples_right(self):
        return self.n_examples_by_label_right.sum(dtype=int)

    def argext(self, arr):
        if self.optimization_mode == 'min':
            extremum = np.argmin
        elif self.optimization_mode == 'max':
            extremum = np.argmax
        extremum_idx = extremum(arr)
        return extremum_idx, arr[extremum_idx]

    def _split_impurity_criterion(self, n_examples_by_label_left, n_examples_by_label_right, n_examples_left, n_examples_right):
        return (self._weighted_impurity_criterion(n_examples_by_label_left, n_examples_left) +
                self._weighted_impurity_criterion(n_examples_by_label_right, n_examples_right)) / \
                (n_examples_left + n_examples_right)

    def _weighted_impurity_criterion(self, n_examples_by_label, n_examples):
        return self.impurity_criterion(n_examples_by_label/n_examples) * n_examples

    def split_makes_gain(self):
        if self.optimization_mode == 'min':
            return self.impurity_score < self.leaf.impurity_score
        elif self.optimization_mode == 'max':
            return self.impurity_score > self.leaf.impurity_score

    def apply_split(self):
        impurity_left = self.impurity_criterion(self.n_examples_by_label_left/self.n_examples_left)
        left_leaf = _DecisionTree(impurity_left,
                                  self.n_examples_by_label_left.copy(),
                                  parent=self.leaf)

        impurity_right = self.impurity_criterion(self.n_examples_by_label_right/self.n_examples_right)
        right_leaf = _DecisionTree(impurity_right,
                                  self.n_examples_by_label_right.copy(),
                                  parent=self.leaf)
        self.leaf.left_subtree = left_leaf
        self.leaf.right_subtree = right_leaf
        self.leaf.rule_threshold = self.rule_threshold
        self.leaf.rule_feature = self.rule_feature
        self.leaf.update_tree()

    def compute_split_X_idx_sorted(self):
        X_idx_sorted_left = np.zeros((self.n_examples_left, self.n_features), dtype=int)
        X_idx_sorted_right = np.zeros((self.n_examples_right, self.n_features), dtype=int)

        left_x_pos = np.zeros(self.n_features, dtype=int)
        right_x_pos = np.zeros(self.n_features, dtype=int)

        for x_idx in self.X_idx_sorted:
            for feat, idx in enumerate(x_idx):
                if self.X[idx, self.rule_feature] < self.rule_threshold:
                    X_idx_sorted_left[left_x_pos[feat], feat] = idx
                    left_x_pos[feat] += 1
                else:
                    X_idx_sorted_right[right_x_pos[feat], feat] = idx
                    right_x_pos[feat] += 1

        return X_idx_sorted_left, X_idx_sorted_right


def gini_impurity_criterion(frac_examples_by_label):
    axis = 1 if len(frac_examples_by_label.shape) > 1 else 0
    return np.sum(frac_examples_by_label * (1 - frac_examples_by_label), axis=axis)

def entropy_impurity_criterion(frac_examples_by_label):
    axis = 1 if len(frac_examples_by_label.shape) > 1 else 0
    return -np.sum(frac_examples_by_label * np.log(frac_examples_by_label), axis=axis)

def margin_impurity_criterion(frac_examples_by_label):
    axis = 1 if len(frac_examples_by_label.shape) > 1 else 0
    return 1 - np.max(frac_examples_by_label, axis=axis)

def breiman_alpha_pruning_objective(tree):
    """
    Taken from Breiman (1980). We want to solve for α in the following equality:
        R(t) + α = R(T_t) + α * n_leaves(T_t),
    where:
        t is the root node of the subtree 'tree'.
        R(t) is the number of errors made by the node t if the node was replaced with a leaf, divided by the total number of examples.
        T_t is the subtree 'tree' taking root at node t
        R(T_t) is the number of errors made by all leaves of the subtree T_t divided by the total number of examples.
        n_leaves(T_t) is the number of leaves of the subtree T_t.
    """
    node_n_errors = tree.n_examples - np.max(tree.n_examples_by_label)
    return (node_n_errors - tree.n_errors) / ( tree.tree_root.n_examples * (tree.n_leaves - 1) )

def leboeuf_alpha_pruning_objective_factory(n_features):
    def leboeuf_alpha_pruning_objective(tree):
        node_n_errors = tree.n_examples - np.max(tree.n_examples_by_label)
        denominator = tree.n_leaves * np.log(tree.n_leaves * n_features) - np.log(n_features)
        return (node_n_errors - tree.n_errors) / (tree.tree_root.n_examples * denominator)
    return leboeuf_alpha_pruning_objective
