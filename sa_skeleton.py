import numpy as np
import matplotlib.pyplot as plt


def sa_skeleton(dim, eval_budget, fitness_func, do_plot=False, return_stats=False):
    """
    skeleton for simulated anealing algorithm. For implementation, we
    recommend to use a vector of integers between [1, n^2] (for magic
    square) as the representation (adapt such a representation to the
    magic cube problem by yourself). In addition, you have to come up
    with the mutation operator by yourself :)

    :param dim:          problem dimension, which should be either 2 (square) or 3 (cube)
    :param eval_budget:  int, the function evaluation budget
    :param fitness_func: function handle, you should use one of the evaluation function provided
    :param do_plot:      should interactive plots be drawn during the optimization
    :param return_stats: should the convergence history be returned too

    :return:
       xopt : array, the final solution vector found by the algorithm
       fopt : double, the corresponding fitness value of xopt

    Author: Koen van der Blom, Hao Wang, Sander van Rijn
    Last modified: 2018-09-28
    """

    # TODO
    # Initialize static parameters
    pm = ...         # mutation rate
    alpha = ...      # temperature decaying parameter
    k = ...          # number of evaluations per iteration
    num_iterations = np.ceil(eval_budget / k)

    # problem size: 12 for square and 7 for cube
    if dim == 2:
        n = 12
    elif dim == 3:
        n = 7
    else:
        raise ValueError('Invalid number of dimensions, use 2 or 3')

    # TODO
    # Set initial temperature
    T = ...

    # Statistics data
    evalcount = 0
    itercount = 0
    fopt = np.inf
    xopt = np.array([np.nan] * n)
    hist_best_f = np.array([np.nan] * eval_budget)
    hist_iter_f = np.array([np.nan] * num_iterations)
    hist_temperature = np.array([np.nan] * num_iterations)

    # TODO
    # Generate initial solution and evaluate
    x = ...
    f = ...         # evaluate the solution using fitness_func

    if f < fopt:
        fopt = f
        xopt = x

    evalcount += 1             # Increase evaluation counter
    hist_best_f[evalcount] = fopt

    if do_plot:
        plt.ion()
        fig = plt.figure()

        ax1 = plt.subplot(131)
        line1, = ax1.plot(hist_best_f[:evalcount])
        ax1.set_title('minimal global error')
        ax1.set_ylabel('error')
        ax1.set_xlabel('evaluations')
        ax1.set_ylim([0, np.max(hist_best_f)])

        ax2 = plt.subplot(132)
        line2, = ax2.plot(np.arange(itercount), hist_temperature[:itercount])
        ax2.set_title('temperature')
        ax2.set_ylabel('T')
        ax2.set_xlabel('iteration')
        ax2.set_ylim([0, T])

        ax3 = plt.subplot(133)
        bar3, = ax2.bar(np.arange(len(xopt)), xopt)
        ax3.set_title('best chromosome')
        ax3.set_ylabel('value')
        ax3.set_xlabel('phenotype index')

        plt.show(block=False)

    # evolution loop
    while evalcount < eval_budget:

        hist_temperature[itercount] = T

        for j in range(k):

            # TODO
            s_new = ...   # Generate a new solution by the permutation of s
            f_new = ...   # evaluate the new solution

            evalcount += 1   # Increase evaluation counter

            if f_new < f:
                # TODO
                # accept the new solution when it is better
                ...
            else:
                # TODO
                # choose to accept or reject the new solution
                # probabilistically based on the current temperature
                ...

            # update the best solution found so far
            if f < fopt:
                fopt = f
                xopt = x

            hist_best_f[evalcount] = fopt   # tracking the best fitness
                                            # ever found

            # Generation best statistics
            hist_iter_f[itercount] = f

            # Plot statistics
            if do_plot:
                line1.set_data(np.arange(evalcount), hist_best_f[:evalcount])
                ax1.set_xlim([0, evalcount])
                ax1.set_ylim([0, np.max(hist_best_f)])

                line2.set_data(np.arange(itercount), hist_temperature[:itercount])
                ax2.set_xlim([0, itercount])

                bar3.set_ydata(xopt)

                plt.draw()


        # TODO
        # Temperature update
        T = ...

        itercount += 1   # Increase iteration counter

    if return_stats:
        return xopt, fopt, hist_best_f
    else:
        return xopt, fopt
