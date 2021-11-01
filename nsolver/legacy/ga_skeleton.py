import numpy as np
import matplotlib.pyplot as plt
from copy import copy


def ga_skeleton(dim, eval_budget, fitness_func, do_plot=False, return_stats=False):
    """
    skeleton for the genetic algorithm. For implementation, we
    recommend to use a vector of integers between [1, n^2] (for magic
    square) as the representation (adapt such a representation to the
    magic cube problem by yourself). In addition, you have to come up
    with the mutation operator by yourself :)

    :param dim:          problem dimension, which should be either 2 (square) or 3 (cube)
    :param eval_budget:  int, the function evaluation budget
    :param fitness_func: function handle, you should use one of the
                         evaluation function provided
    :param do_plot:      should interactive plots be drawn during the optimization
    :param return_stats: should the convergence history be returned too

    :return:
        xopt : array, the final solution vector found by the algorithm
        fopt : double, the corresponding fitness value of xopt

    Author: Koen van der Blom, Hao Wang, Sander van Rijn
    Last modified: 2018-09-28
    """

    # ----------------- general setting of variables ----------------------

    # TODO
    # static variables
    # problem size: 12 for square and 7 for cube
    if dim == 2:
        n = 12
    elif dim == 3:
        n = 7
    else:
        raise ValueError('Invalid number of dimensions, use 2 or 3')

    # the pheno type of the solution is the permutation of integers
    # from 1 to n ^ dim
    pheno_len = n**dim

    # TODO
    # At this point, you should think which geno type representation you
    # would like to use and thus determine the length of the geno type
    # solution vector. Example of geno encoding: bit-string converted from
    # array of integers. This is up to you :)
    geno_len = ...

    # TODO
    # endogenous parameters setting
    mu = ...               # population size
    pc = ...               # crossover rate
    pm = ...               # mutation rate

    # internal counter variable
    evalcount = 0     # count function evaluations
    gencount = 0      # count generation/iterations

    # historical information
    hist_best_f = np.zeros(eval_budget)
    hist_gen_f = np.zeros(np.ceil(eval_budget/mu))


    # ------------------- population initialization -----------------------
    # row vector representation is used throughout this code
    # you need to keep pheno type population updated with the geno types
    # for function evaluation

    # population
    pop_pheno = np.zeros((mu, pheno_len))   # pheno type
    pop_geno = np.zeros((mu, geno_len))     # geno type
    fitness = np.zeros(mu)                  # fitness values

    # TODO
    for i in range(mu):
        pop_pheno[i, :] = ...  # generate pheno type individual uniformly
        pop_geno[i, :] = ...   # convert them to geno type solution
        fitness[i] = fitness_func(pop_pheno[i, :])   # and evaluate the
                                                     # solution...

    index = np.argmin(fitness)
    fopt = fitness[index]
    xopt = copy(pop_geno[index, :])
    xopt_pheno = copy(pop_pheno[index, :])

    # increase the evalcount by mu
    hist_best_f[evalcount:evalcount+mu] = fopt
    evalcount += mu

    if do_plot:
        plt.ion()
        fig = plt.figure()

        ax1 = plt.subplot(131)
        line1 = ax1.plot(hist_best_f[:evalcount])[0]
        ax1.set_title('minimal global error')
        ax1.set_ylabel('error')
        ax1.set_xlabel('evaluations')
        ax1.set_ylim([0, np.max(hist_best_f)])

        ax2 = plt.subplot(132)
        line2 = ax2.plot(hist_gen_f[:gencount])[0]
        ax2.set_title('minimal error in the current generation')
        ax2.set_ylabel('error')
        ax2.set_xlabel('generation')
        ax2.set_ylim([0, np.max(hist_gen_f)])

        ax3 = plt.subplot(133)
        bars3 = ax3.bar(np.arange(pheno_len), xopt_pheno)
        ax3.set_title('best chromosome')
        ax3.set_ylabel('value')
        ax3.set_xlabel('phenotype index')

        plt.show(block=False)

    # ----------------------- Evolution loop ------------------------------
    while evalcount < eval_budget:

        # generate the a new population using crossover and mutation
        pop_new_geno = np.zeros((mu, geno_len))
        for i in range(mu):

            # TODO
            # implement the selection operator.
            p1 = ...               # select the first parent from pop_geno
            if np.random.randn() < pc:
                p2 = ...           # select the second parent from pop_geno

                # TODO
                # implement the crossover operator
                ...     # crossover p1 and p2

            else:

                # No crossover, copy the parent chromosome
                ...


            # TODO
            # implement the muation operator
            pop_new_geno[i, :] = ...         # apply the mutation and then
                                             # store it in pop_new_geno

            # TODO
            # repair the newly generated solution (if you want...)
            # the solution might be invalid because of duplicated integers
            ...


        # Replace old population by the newly generated population
        pop_geno = pop_new_geno

        # TODO
        for i in range(mu):
            pop_pheno[i, :] = ...   # decode the geno type solution to
                                    # pheno type for evaluation
            fitness[i] = fitness_func(pop_pheno[i, :])

        # optimal solution in each iteration
        index = np.argmin(fitness)
        x_opt_curr_gen = copy(pop_geno[index, :])
        x_opt_pheno_curr_gen = copy(pop_pheno[index, :])
        fopt_curr_gen = fitness[index]

        # keep track of the best solution ever found
        if fopt_curr_gen < fopt:
            fopt = fopt_curr_gen
            xopt = copy(x_opt_curr_gen)
            xopt_pheno = copy(x_opt_pheno_curr_gen)

        # record historical information
        hist_best_f[evalcount:evalcount+mu] = fopt
        hist_gen_f[gencount] = fopt_curr_gen

        # internal counters increment
        gencount += 1
        evalcount += mu

        # Plot statistics
        if do_plot:
            line1.set_data(np.arange(evalcount), hist_best_f[:evalcount])
            ax1.set_xlim([0, evalcount])
            ax1.set_ylim([0, np.max(hist_best_f)])

            line2.set_data(np.arange(gencount), hist_gen_f[:gencount])
            ax2.set_xlim([0, gencount])
            ax2.set_ylim([0, np.max(hist_gen_f)])

            for bar, h in zip(bars3, xopt):
                bar.set_height(h)

            plt.pause(0.00001)
            plt.draw()

    if return_stats:
        return xopt, fopt, hist_best_f
    else:
        return xopt, fopt
