import numpy as np
from copy import copy
from scipy.special import zeta, gammaln
from partitioning_machines import growth_function_upper_bound
from partitioning_machines import wedderburn_etherington


def shawe_taylor_bound(n_examples,
                       n_errors,
                       growth_function,
                       errors_logprob,
                       complexity_logprob,
                       delta=.05,
                       ):
    """
    Theorem 2.3 of Shawe-Taylor et al. (1997), Structural Risk Minimization over Data-Dependent Hierarchies, with the modification that Sauer's lemma is not used.
    """
    epsilon = 2*n_errors + 4*(np.log(float(growth_function(2*n_examples)))
                              + np.log(4)
                              - np.log(delta)
                              - errors_logprob
                              - complexity_logprob)
    return epsilon / n_examples

def shawe_taylor_bound_pruning_objective_factory(n_features,
                                                 table=dict(),
                                                 loose_pfub=True,
                                                 errors_logprob_prior=None,
                                                 complexity_logprob_prior=None,
                                                 delta=.05):
    if errors_logprob_prior is None:
        r = 1/2
        errors_logprob_prior = lambda n_errors: np.log(1-r) + n_errors*np.log(r)

    if complexity_logprob_prior is None:
        s = 2
        complexity_logprob_prior = lambda complexity_idx: -np.log(zeta(s)) - s*np.log(complexity_idx) - np.log(float(wedderburn_etherington(complexity_idx)))

    def shawe_taylor_bound_pruning_objective(subtree):
        copy_of_tree = copy(subtree.root)
        copy_of_subtree = copy_of_tree.follow_path(subtree.path_from_root())
        copy_of_subtree.remove_subtree()

        n_classes = copy_of_tree.n_examples_by_label.shape[0]
        growth_function = growth_function_upper_bound(copy_of_tree, n_features, n_classes, table, loose_pfub)
        n_examples = copy_of_tree.n_examples
        n_errors = copy_of_tree.n_errors
        errors_logprob = errors_logprob_prior(n_errors)
        complexity_logprob = complexity_logprob_prior(copy_of_tree.n_leaves)

        return shawe_taylor_bound(n_examples, n_errors, growth_function, errors_logprob, complexity_logprob, delta)

    return shawe_taylor_bound_pruning_objective


def vapnik_bound(n_examples,
                 n_errors,
                 growth_function,
                 errors_logprob,
                 complexity_logprob,
                 delta=.05,
                 ):
    """
    Equation (4.41) of Vapnik's book (1998) extended to SRM.
    """
    epsilon = 4 / n_examples * (np.log(float(growth_function(2*n_examples)))
                                + np.log(4)
                                - np.log(delta)
                                - errors_logprob
                                - complexity_logprob)

    empirical_risk = n_errors / n_examples

    return empirical_risk + epsilon/2 * (1 + np.sqrt(1 + 4*empirical_risk/epsilon))

def vapnik_bound_pruning_objective_factory(n_features,
                                           table=dict(),
                                           loose_pfub=True,
                                           errors_logprob_prior=None,
                                           complexity_logprob_prior=None,
                                           delta=.05):
    if errors_logprob_prior is None:
        r = 1/2
        errors_logprob_prior = lambda n_errors: np.log(1-r) + n_errors*np.log(r)

    if complexity_logprob_prior is None:
        s = 2
        complexity_logprob_prior = lambda complexity_idx: -np.log(zeta(s)) - s*np.log(complexity_idx) - np.log(float(wedderburn_etherington(complexity_idx)))

    def vapnik_bound_pruning_objective(subtree):
        copy_of_tree = copy(subtree.root)
        copy_of_subtree = copy_of_tree.follow_path(subtree.path_from_root())
        copy_of_subtree.remove_subtree()

        n_classes = copy_of_tree.n_examples_by_label.shape[0]
        growth_function = growth_function_upper_bound(copy_of_tree, n_features, n_classes, table, loose_pfub)
        n_examples = copy_of_tree.n_examples
        n_errors = copy_of_tree.n_errors
        errors_logprob = errors_logprob_prior(n_errors)
        complexity_logprob = complexity_logprob_prior(copy_of_tree.n_leaves)

        return vapnik_bound(n_examples, n_errors, growth_function, errors_logprob, complexity_logprob, delta)
    return vapnik_bound_pruning_objective


def binomln(n, k):
    """
    Computes the logarithmic binomial coefficients (to avoid overflows).
    """
    return gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)

def single_term_of_alternate_cdf(k, K, m, M):
    """
    Berkopec presents an alternate expression for the CDF of the hypergeometric distribution. This function computes one term of the sum.
    """
    return np.exp(binomln(K, k) + binomln(M-1-K, M-m+k-K) - binomln(M, m))

def hyper_inv(n_errors, delta, n_examples, total_examples):
    """
    Computes pseudo inverse of the cumultative hypergeometric distribution on parameter 'total_errors'.

    Returns:
        min{ i : hyp(n_errors, i, n_examples, total_examples) <= delta }
    """
    total_errors = total_examples - n_examples + n_errors
    cumul = single_term_of_alternate_cdf(n_errors, total_errors, n_examples, total_examples)
    while (cumul <= delta) and total_errors >= n_errors:
        total_errors -= 1
        cumul += single_term_of_alternate_cdf(n_errors, total_errors, n_examples, total_examples)

    return total_errors + 1

def hyper_inv_bound(n_examples,
                    n_errors,
                    growth_function,
                    complexity_prob,
                    ghost_sample_size=None,
                    delta=.05,
                    ):
    """
    """
    if ghost_sample_size is None:
        ghost_sample_size = 4*n_examples

    epsilon = 1/ghost_sample_size * max(1, hyper_inv(n_errors, delta*complexity_prob/growth_function(n_examples+ghost_sample_size), n_examples, ghost_sample_size+n_examples)-1-n_errors )

    return epsilon

def hyper_inv_bound_pruning_objective_factory(n_features,
                                              table=dict(),
                                              loose_pfub=True,
                                              complexity_prob_prior=None,
                                              ghost_sample_size=None,
                                              delta=.05):
    if complexity_prob_prior is None:
        s = 2
        complexity_prob_prior = lambda complexity_idx: 1/(zeta(s) * complexity_idx**s *float(wedderburn_etherington(complexity_idx) ) )

    def hyper_inv_bound_pruning_objective(subtree):
        copy_of_tree = copy(subtree.root)
        copy_of_subtree = copy_of_tree.follow_path(subtree.path_from_root())
        copy_of_subtree.remove_subtree()

        n_classes = copy_of_tree.n_examples_by_label.shape[0]
        growth_function = growth_function_upper_bound(copy_of_tree, n_features, n_classes, table, loose_pfub)
        n_examples = copy_of_tree.n_examples
        n_errors = copy_of_tree.n_errors
        complexity_prob = complexity_prob_prior(copy_of_tree.n_leaves)

        return hyper_inv_bound(n_examples, n_errors, growth_function, complexity_prob, ghost_sample_size, delta)

    return hyper_inv_bound_pruning_objective