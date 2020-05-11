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
        self.label = np.eye(1, n_examples_by_label.shape[0], k=np.argmax(self.n_examples_by_label))
        self.rule_threshold = rule_threshold
        self.rule_feature = rule_feature
    
    def predict(self, x):
        if self.is_leaf():
            return self.label
        else:
            if x[self.rule_feature] < self.rule_threshold:
                return self.left_subtree.predict(x)
            else:
                return self.right_subtree.predict(x)
    
    def is_pure(self):
        return (self.n_examples_by_label == self.n_examples_by_label.sum()).any()


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
        
        possible_splits = [Splitter(self.tree, X, encoded_y, X_idx_sorted, self.impurity_criterion, self.optimization_mode, self.min_examples_per_leaf)] # List of splits that can be produced.
        c=0
        while possible_splits and self.tree.n_leaves < self.max_n_leaves:
            c+=1
            best_split = possible_splits[0]
            for split in possible_splits:
                if self.optimization_mode == 'min':
                    if best_split.impurity_score > split.impurity_score:
                        best_split = split
                elif self.optimization_mode == 'max':
                    if best_split.impurity_score < split.impurity_score:
                        best_split = split

            best_split.apply_split()
            
            if self.tree.n_leaves < self.max_n_leaves and self.tree.height < self.max_depth:
                if best_split.split_makes_gain():
                    possible_splits.extend(best_split.leaves_splitters())
                    
            possible_splits.remove(best_split)
        
        return self
        
    def _init_tree(self, encoded_y, n_examples):
        n_examples_by_label = np.sum(encoded_y, axis=0)
        self.tree = _DecisionTree(self.impurity_criterion(n_examples_by_label/n_examples),
                         n_examples_by_label)
        
    def predict(self, X):
        return self.label_encoder.decode_labels(np.array([self.tree.predict(x) for x in X]))

    def predict_proba(self, X):
        pass
    


class Splitter:
    def __init__(self, leaf, X, y, X_idx_sorted, impurity_criterion, optimization_mode, min_examples_per_leaf=1):
        self.leaf = leaf
        self.X = X
        self.y = y
        self.X_idx_sorted = X_idx_sorted
        self.n_examples, self.n_features = X.shape
        self.impurity_criterion = impurity_criterion
        self.optimization_mode = optimization_mode
        self.min_examples_per_leaf = min_examples_per_leaf
        
        self._find_best_split()
    
    def _find_best_split(self):
        n_examples_by_label = self.leaf.n_examples_by_label
        if self.leaf.is_pure():
            return
        
        if self.n_examples < 2*self.min_examples_per_leaf:
            return
        
        n_examples_left = self.min_examples_per_leaf
        n_examples_right = self.n_examples - n_examples_left
        
        n_examples_by_label_left = sum(self.y[self.X_idx_sorted[i]] for i in range(n_examples_left)) # Shape: (n_features, n_classes)
        n_examples_by_label_right = n_examples_by_label - n_examples_by_label_left

        self.rule_feature, self.impurity_score = self.argext(self._split_impurity_criterion(n_examples_by_label_left, n_examples_by_label_right, n_examples_left, n_examples_right))
        rule_threshold_idx_left = self.X_idx_sorted[n_examples_left-1, self.rule_feature]
        rule_threshold_idx_right = self.X_idx_sorted[n_examples_left, self.rule_feature]
        
        self.n_examples_by_label_left = n_examples_by_label_left[self.rule_feature].copy()
        self.n_examples_by_label_right = n_examples_by_label_right[self.rule_feature].copy()

        for row, x_idx in enumerate(self.X_idx_sorted[self.min_examples_per_leaf:-self.min_examples_per_leaf]):
            n_examples_left += 1
            n_examples_right -= 1
            transfered_labels = self.y[x_idx]
            n_examples_by_label_left += transfered_labels
            n_examples_by_label_right -= transfered_labels
            tmp_feature, tmp_impurity_score = self.argext(self._split_impurity_criterion(n_examples_by_label_left, n_examples_by_label_right, n_examples_left, n_examples_right))
            if (self.optimization_mode == 'min' and tmp_impurity_score < self.impurity_score) \
                or (self.optimization_mode == 'max' and tmp_impurity_score > self.impurity_score):
                    self.impurity_score = tmp_impurity_score
                    self.rule_feature = tmp_feature
                    rule_threshold_idx_left = x_idx[self.rule_feature]
                    rule_threshold_idx_right = self.X_idx_sorted[row+2, self.rule_feature]
                    self.n_examples_by_label_left = n_examples_by_label_left[self.rule_feature].copy()
                    self.n_examples_by_label_right = n_examples_by_label_right[self.rule_feature].copy()
        
        self.rule_threshold = (self.X[rule_threshold_idx_left, self.rule_feature] + self.X[rule_threshold_idx_right, self.rule_feature])/2
    
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
        left_leaf = _DecisionTree(impurity_left, self.n_examples_by_label_left, parent=self.leaf)
        
        impurity_right = self.impurity_criterion(self.n_examples_by_label_right/self.n_examples_right)
        right_leaf = _DecisionTree(impurity_right, self.n_examples_by_label_right, parent=self.leaf)
        self.leaf.left_subtree = left_leaf
        self.leaf.right_subtree = right_leaf
        self.leaf.rule_threshold = self.rule_threshold
        self.leaf.rule_feature = self.rule_feature
        self.leaf.update_tree()
    
    def leaves_splitters(self, left=True, right=True):
        X_idx_sorted_left, X_idx_sorted_right = self._compute_split_X_idx_sorted()
        splitters = []
        
        if left:
            left_splitter = type(self)(self.leaf.left_subtree, self.X, self.y, X_idx_sorted_left, self.impurity_criterion, self.optimization_mode)
            if not left_splitter.leaf.is_pure():
                splitters.append(left_splitter)
        
        if right:
            right_splitter = type(self)(self.leaf.right_subtree, self.X, self.y, X_idx_sorted_left, self.impurity_criterion, self.optimization_mode)
            if not right_splitter.leaf.is_pure():
                splitters.append(right_splitter)

        return splitters
        
    def _compute_split_X_idx_sorted(self):
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