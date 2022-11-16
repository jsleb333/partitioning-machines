# Partitioning Machines: A Framework to characterize the VC dimension of decision trees

## Preface
This package provides an implementation of the algorithms presented in the work of Leboeuf, LeBlanc and Marchand (2022) named "Generalization Properties of Decision Trees on Real-valued and Categorical Features".

## Content
This package provides implementations of the algorithms that computes the bounds on the partitioning functions, the growth function and the VC dimension of binary decision trees, as well as other useful tools to work with scikit-learn decision trees.
The file 'main.py' contains a usage example to compute the VC dimension bounds on the first 11 non-equivalent trees.

### Detailled content
- Tree object built in a recursive manner in the file 'tree.py'.
- Decision tree classifier that grows a tree following CART's algorithm in 'decision_tree_classifier.py'.
- Partitioning function upper bound object that implements an optimized version of the algorithm 2 of Appendix E in the file 'partitioning_function_upper_bound.py'.
- VC dimension lower and upper bounds algorithms provided in the file 'vcdim.py'.
- Tree conversion from scikit-learn decision trees to present implementation in the file 'utils.py'.
