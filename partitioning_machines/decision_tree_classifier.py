import numpy as np

from partitioning_machines import Tree


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
        
        self.label_decoding = list(sorted(set(y_i for y_i in y)))
        self.label_encoding = {y_i:i for i, y_i in enumerate(self.label_decoding)}
        self.n_classes = len(self.label_decoding)
        encoded_y = np.array([self.label_encoding[y_i] for y_i in y])
        
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
        proportion_of_observations = np.array([])
    
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
    