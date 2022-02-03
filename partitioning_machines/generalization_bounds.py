import numpy as np

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
