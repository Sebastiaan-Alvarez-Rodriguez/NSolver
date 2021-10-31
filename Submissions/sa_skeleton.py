import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def random_solution(n, dim):
    return np.random.permutation(np.arange(1, (n ** dim)+1))
    
def mutate_solution(s, n, dim, pm, f, fopt):
    length = n ** dim
    s_cpy = copy(s)
    
    # Do more mutations if we are far away from the optimal solution we found
    diff = max(f - fopt, 1)
    t_evals = int(np.ceil(diff * pm))
    
    # Mutate n times by swapping
    for i in range(1, t_evals):
        mut_left = np.random.randint(0, high=length-1)
        mut_right = np.random.randint(mut_left, high=length)
        s_cpy[mut_left], s_cpy[mut_right] = s_cpy[mut_right], s_cpy[mut_left]
    return s_cpy

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

    # Initialize static parameters
    pm = 2         # mutation rate
    alpha = 0.7      # temperature decaying parameter
    k = 100          # number of evaluations per iteration
    num_iterations = int(np.ceil(eval_budget / k))

    # problem size: 12 for square and 7 for cube
    if dim == 2:
        n = 12
    elif dim == 3:
        n = 7
    else:
        raise ValueError('Invalid number of dimensions, use 2 or 3')

    # Set initial temperature
    T = 25000

    # Statistics data
    evalcount = 0
    itercount = 0
    fopt = np.inf
    xopt = np.array([np.nan] * n**dim)
    hist_best_f = np.array([np.nan] * eval_budget)
    hist_iter_f = np.array([np.nan] * num_iterations)
    hist_temperature = np.array([np.nan] * num_iterations)

    # Generate initial solution and evaluate
    x = random_solution(n, dim)
    f = fitness_func(x)         # evaluate the solution using fitness_func

    if f < fopt:
        fopt = f
        xopt = x

    hist_best_f[evalcount] = fopt
    evalcount += 1             # Increase evaluation counter

    if do_plot:
        plt.ion()
        fig = plt.figure()

        ax1 = plt.subplot(131)
        line1 = ax1.plot(hist_best_f[:evalcount])[0]
        ax1.set_title('minimal global error')
        ax1.set_ylabel('error')
        ax1.set_xlabel('evaluations')
        ax1.set_ylim([0, np.max(hist_best_f[:evalcount])])

        ax2 = plt.subplot(132)
        line2 = ax2.plot(np.arange(itercount), hist_temperature[:itercount])[0]
        ax2.set_title('temperature')
        ax2.set_ylabel('T')
        ax2.set_xlabel('iteration')
        ax2.set_ylim([0, T])

        ax3 = plt.subplot(133)
        bars3 = ax3.bar(np.arange(len(xopt)), xopt)
        ax3.set_title('best representation')
        ax3.set_ylabel('value')
        ax3.set_xlabel('representation index')

        plt.show(block=False)
    
    # evolution loop
    while evalcount < eval_budget:
        
        hist_temperature[itercount] = T
        
        k = min(k, eval_budget-evalcount)
        for j in range(k):

            s_new = mutate_solution(x, n, dim, pm, f, fopt)   # Generate a new solution by the permutation of s
            f_new = fitness_func(s_new)   # evaluate the new solution
            
            if f_new < f:
                x = s_new
                f = f_new
            elif np.random.randn() < np.exp(-(f_new - f) / T):
                x = s_new
                f = f_new
                
            # Reset back to the original if we are to far away from optimal
            if f > 2 * fopt:
                x = copy(xopt)
                f = fopt
            
            # update the best solution found so far
            if f < fopt:
                fopt = f
                xopt = copy(x)
            
            hist_best_f[evalcount] = fopt   # tracking the best fitness ever found
            
            # Generation best statistics
            hist_iter_f[itercount] = f
            
            # Plot statistics
            if do_plot:
                line1.set_data(np.arange(evalcount), hist_best_f[:evalcount])
                ax1.set_xlim([0, evalcount])
                ax1.set_ylim([0, np.max(hist_best_f[:evalcount])])

                line2.set_data(np.arange(itercount), hist_temperature[:itercount])
                ax2.set_xlim([0, itercount])

                for bar, h in zip(bars3, xopt):
                    bar.set_height(h)

                plt.pause(0.00001)
                plt.draw()
            evalcount += 1   # Increase evaluation counter
        T = alpha * T
        
        print(evalcount, ": current fitness: ", fopt)
        itercount += 1   # Increase iteration counter

    if return_stats:
        return xopt, fopt, hist_best_f
    else:
        return xopt, fopt
