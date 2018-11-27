#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

"""
Filename.py: << A short summary docstring about this file >>
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


def reference_random(dim, eval_budget, fitness_func, do_plot=False, return_stats=False):
    """A reference random search implementation"""

    if dim == 2:
        n = 12
    elif dim == 3:
        n = 7
    else:
        raise ValueError('Invalid number of dimensions, use 2 or 3')

    length = n**dim
    xopt = None
    fopt = np.inf
    hist_best_f = np.zeros(eval_budget)
    evals_used = 0

    while fopt > 0 and evals_used < eval_budget:
        x = np.random.permutation(length)+1
        f = fitness_func(x)

        if f < fopt:
            fopt = f
            xopt = x

        hist_best_f[evals_used] = fopt
        evals_used += 1

    if return_stats:
        return xopt, fopt, hist_best_f
    else:
        return xopt, fopt
