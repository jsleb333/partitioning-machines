import numpy as np

from partitioning_machines import Tree, OneHotEncoder


class DecisionTreeClassifier:
    def __init__(self,
                 criterion=None,
                 optimization_mode='min',
                 max_n_leaves=None,
                 max_depth=None,
                 min_examples_per_leaf=1):
        self.criterion = criterion
        self.optimization_mode = optimization_mode
        self.max_n_leaves = max_n_leaves
        self.max_depth = max_depth
        self.tree = Tree()
        
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
                    if best_split.criterion_score > split.criterion_score:
                        best_split = split
                elif self.optimization_mode == 'max':
                    if best_split.criterion_score < split.criterion_score:
                        best_split = split

            best_split.leaf.replace_subtree(best_split.stump)
            
            if self.tree.n_leaves < self.max_n_leaves and self.tree.depth < self.max_depth:
                possible_splits.append(best_split.split_left_leaf())
                possible_splits.append(best_split.split_right_leaf())
            
            possible_splits.remove(best_split)

    def predict(self, X, X_idx_sorted=None):
        pass

    def predict_proba(self, X, X_idx_sorted=None):
        pass
    


class Split:
    def __init__(self, leaf, X, y, X_idx_sorted):
        self.leaf = leaf
        self.X = X
        self.y = y
        self.X_idx_sorted = X_idx_sorted
        self.n_examples, self.n_features = X.shape
    
    def _find_best_split(self):
        proportion_of_observations_left = np.sum(self.y[1:], axis=0)/(self.n_examples-1)
    
    @property
    def stump(self):
        raise NotImplementedError
    
    @property
    def criterion_score(self):
        raise NotImplementedError
    
    def split_left_leaf(self):
        pass

    def split_right_leaf(self):
        pass


def gini_impurity_criterion(proportion_of_observations):
    return np.sum(proportion_of_observations * (1 - proportion_of_observations))

def entropy_impurity_criterion(proportion_of_observations):
    return -np.sum(proportion_of_observations * np.log(proportion_of_observations))

def margin_impurity_criterion(proportion_of_observations):
    return 1 - np.max(proportion_of_observations)