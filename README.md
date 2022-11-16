# Partitioning Machines: A Framework to characterize the VC dimension of decision trees

## Preface
This package provides an implementation of the algorithms presented in the work of Leboeuf, LeBlanc and Marchand (2020) named "Decision trees as partitioning machines to characterize their generalization properties".

## Content

This package provides implementations of the algorithms that computes an upper and lower bound on the VC dimension of binary decision trees, as well as other useful tools to work with scikit-learn decision trees.
The file 'main.py' contains a usage example to compute the VC dimension bounds on the first 11 non-equivalent trees.

### Detailled content
- Tree object built in a recursive manner in the file 'tree.py'.
- Partitioning function upper bound object that implements an optimized version of the algorithm 1 of Appendix E in the file 'partitioning_function_upper_bound.py'.
- VC dimension lower and upper bounds algorithms provided in the file 'vcdim.py'.
- Tree conversion from scikit-learn decision trees to present implementation in the file 'utils.py'.
