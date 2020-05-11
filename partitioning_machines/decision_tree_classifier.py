import numpy as np

from partitioning_machines import Tree as _Tree
from partitioning_machines import OneHotEncoder


class Tree(_Tree):
    def __init__(self,
                 impurity_score,
                 n_examples_by_label,
                 *args,
                 rule_threshold=None,
                 rule_feature=None,
                 **kwargs):
        super().__init__(left_subtree=left_subtree,
                         right_subtree=right_subtree,
                         parent=parent,
                         height=height,
                         depth=depth,
                         n_leaves=n_leaves,
                         n_nodes=n_nodes,
                         hash_value=hash_value,
                         position=position)
        self.impurity_score = impurity_score
        self.n_examples_by_label = n_examples_by_label
        self.label = np.eye(n_examples_by_label.shape[0], k=np.argmax(self.n_examples_by_label))
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


class DecisionTreeClassifier:
    def __init__(self,
                 impurity_criterion=None,
                 optimization_mode='min',
                 max_n_leaves=None,
                 max_depth=None,
                 min_examples_per_leaf=1):
        self.impurity_criterion = impurity_criterion
        self.optimization_mode = optimization_mode
        self.max_n_leaves = max_n_leaves
        self.max_depth = max_depth
        self.tree = None
        
    def fit(self, X, y, X_idx_sorted=None):
        n_examples, n_features = X.shape
        self.label_encoder = OneHotEncoder(y)
        encoded_y = self.label_encoder.encode_labels(y)

        if self.max_n_leaves is None:
            self.max_n_leaves = n_examples
        
        possible_splits = [Split(self.tree, X, encoded_y, X_idx_sorted)] # List of splits that can be produced.
        
        while possible_splits and self.tree.n_leaves < self.max_n_leaves:
            best_split = possible_splits[0]
            for split in possible_splits:
                if self.optimization_mode == 'min':
                    if best_split.impurity_score > split.impurity_score:
                        best_split = split
                elif self.optimization_mode == 'max':
                    if best_split.impurity_score < split.impurity_score:
                        best_split = split

            best_split.leaf.replace_subtree(best_split.stump)
            
            if self.tree.n_leaves < self.max_n_leaves and self.tree.depth < self.max_depth:
                if best_split.split_makes_gain():
                    possible_splits.append(best_split.split_left_leaf())
                    possible_splits.append(best_split.split_right_leaf())
            
            possible_splits.remove(best_split)

    def _init_tree(self, n_examples, encoded_y):
        n_examples_by_label = np.sum(encoded_y, axis=0)
        self.tree.n_examples_by_label = n_examples_by_label
        self.tree.impurity_score = self.impurity_criterion(n_examples_by_label/n_examples)

    def predict(self, X):
        return self.label_encoder.decode_labels(np.array([self.tree.predict(x) for x in X]))

    def predict_proba(self, X):
        pass
    


class Split:
    def __init__(self, leaf, X, y, X_idx_sorted, impurity_criterion, optimization_mode):
        self.leaf = leaf
        self.X = X
        self.y = y
        self.X_idx_sorted = X_idx_sorted
        self.n_examples, self.n_features = X.shape
        self.impurity_criterion = impurity_criterion
        self.optimization_mode = optimization_mode
    
    def _find_best_split(self):
        n_examples_by_label = self.leaf.n_examples_by_label
        n_examples_left = 1
        n_examples_right = self.n_examples - n_examples_left
        
        n_examples_by_label_left = self.y[self.X_idx_sorted[0]] # Shape: (n_classes, n_features)
        n_examples_by_label_right = n_examples_by_label - n_examples_by_label_left
        

        self.rule_feature, self.impurity_score = self.argext(self._split_impurity_criterion(n_examples_by_label_left, n_examples_by_label_right, n_examples_left, n_examples_right))
        self.rule_threshold_idx = self.X_idx_sorted[0, self.rule_feature]
        
        self.n_examples_by_label_left = n_examples_by_label_left[:,self.rule_feature].copy()
        self.n_examples_by_label_right = n_examples_by_label_right[:,self.rule_feature].copy()

        for x_idx in self.X_idx_sorted[1:-1]:
            n_examples_left += 1
            n_examples_right -= 1
            transfered_labels = self.y[x_idx]
            n_examples_by_label_left += transfered_labels
            n_examples_by_label_right -= transfered_labels
            tmp_feature, tmp_impurity_score = self.argext(self._split_impurity_criterion(n_examples_by_label_left, n_examples_by_label_right, n_examples_left, n_examples_right))
            if self.optimization_mode == 'min':
                if tmp_impurity_score < self.impurity_score:
                    self.impurity_score = tmp_impurity_score
                    self.rule_feature = tmp_feature
                    self.rule_threshold_idx = x_idx[self.rule_feature]
                    self.n_examples_left = n_examples_left
                    self.n_examples_right = n_examples_right
                    self.n_examples_by_label_left = n_examples_by_label_left[:,self.rule_feature].copy()
                    self.n_examples_by_label_right = n_examples_by_label_right[:,self.rule_feature].copy()
            elif self.optimization_mode == 'max':
                if tmp_impurity_score > self.impurity_score:
                    self.impurity_score = tmp_impurity_score
                    self.rule_feature = tmp_feature
                    self.rule_threshold_idx = x_idx[self.rule_feature]
                    self.n_examples_left = n_examples_left
                    self.n_examples_right = n_examples_right
                    self.n_examples_by_label_left = n_examples_by_label_left[:,self.rule_feature].copy()
                    self.n_examples_by_label_right = n_examples_by_label_right[:,self.rule_feature].copy()
        
        self.rule_threshold = (self.X[self.best_threshold_idx, self.rule_feature] + self.X[self.best_threshold_idx+1, self.rule_feature])/2
    
    def _compute_split_X_idx_sorted(self):
        self.X_idx_sorted_left = np.zeros((self.n_examples_left, self.n_features))
        self.X_idx_sorted_right = np.zeros((self.n_examples_right, self.n_features))

        left_x_pos = 0
        right_x_pos = 0
        for x_idx in self.X_idx_sorted:
            for feat, idx in enumerate(x_idx):
                if idx in self.X_idx_sorted[:self.rule_threshold_idx, self.rule_feature]:
                    self.X_idx_sorted_left[left_x_pos, feat] = idx
                    left_x_pos += 1
                else:
                    self.X_idx_sorted_right[right_x_pos, feat] = idx
                    right_x_pos += 1
    
    def _weighted_impurity_criterion(self, n_examples_by_label, n_examples):
        return self.impurity_criterion(n_examples_by_label/n_examples) * n_examples

    def _split_impurity_criterion(self, n_examples_by_label_left, n_examples_by_label_right, n_examples_left, n_examples_right):
        return (self._weighted_impurity_criterion(n_examples_by_label_left, n_examples_left) + 
                self._weighted_impurity_criterion(n_examples_by_label_left, n_examples_left)) / \
                (n_examples_left + n_examples_right)

    def argext(self, arr):
        if self.optimization_mode == 'min':
            extremum = np.argmin
        elif self.optimization_mode == 'max':
            extremum = np.argmax
        
        extremum_idx = extremum(arr)
        
        return extremum_idx, arr(extremum_idx)
    
    def split_makes_gain(self):
        if self.optimization_mode == 'min':
            return self.impurity_score < self.leaf.impurity_score
        elif self.optimization_mode == 'max':
            return self.impurity_score > self.leaf.impurity_score
            
    @property
    def stump(self):
        left_leaf = Tree()
        left_leaf.n_examples_by_label = self.n_examples_by_label_left
        return 
    
    def split_left_leaf(self):
        pass

    def split_right_leaf(self):
        pass


def gini_impurity_criterion(frac_examples_by_label):
    return np.sum(frac_examples_by_label * (1 - frac_examples_by_label))

def entropy_impurity_criterion(frac_examples_by_label):
    return -np.sum(frac_examples_by_label * np.log(frac_examples_by_label))

def margin_impurity_criterion(frac_examples_by_label):
    return 1 - np.max(frac_examples_by_label)