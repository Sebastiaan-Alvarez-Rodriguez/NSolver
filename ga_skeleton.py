import numpy as np
import matplotlib.pyplot as plt

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

    if dim == 2:
        n = 12
    elif dim == 3:
        n = 7
    else:
        raise ValueError('Invalid number of dimensions, use 2 or 3')

    #Een array met ints, waarbij elke int een getal uit het antwoord representeert
    geno_len = n**dim 

    # TODO
    # endogenous parameters setting
    mu = n**dim          # population size
    pc = 10              # crossover rate
    pm = 10              # mutation rate

    # internal counter variable
    evalcount = 0     # count function evaluations
    gencount = 0      # count generation/iterations

    # historical information
    hist_best_f = np.zeros(eval_budget)
    hist_gen_f = np.zeros(int(np.ceil(eval_budget/mu)))


    # ------------------- population initialization -----------------------
    # row vector representation is used throughout this code
    fitness = np.zeros((mu))
    # population
    pop_geno = np.zeros((mu, geno_len))

    # TODO (now initializes all members of population a random value [1, 144])
    for i in range(mu):
        pop_geno[i, :] = np.random.permutation(np.arange(1, geno_len+1))
        fitness[i] = fitness_func(pop_geno[i, :])   # and evaluate the solution...
    print('All', mu, 'population members initialized')
    index = np.argmin(fitness)
    fopt = fitness[index]
    xopt = pop_geno[index, :]

    # increase the evalcount by mu
    hist_best_f[evalcount:evalcount+mu] = fopt
    evalcount += mu

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
        line2, = ax1.plot(hist_gen_f[:gencount])
        ax2.set_title('minimal error in the current generation')
        ax2.set_ylabel('error')
        ax2.set_xlabel('generation')
        ax2.set_ylim([0, np.max(hist_gen_f)])

        plt.show(block=False)

    # ----------------------- Evolution loop ------------------------------
#https://blackboard.leidenuniv.nl/bbcswebdav/pid-4458530-dt-content-rid-5832938_1/courses/4032NACO6-1819FWN/3b%20-%20Evolutionary_Algorithms.pdf
    print('Starting evolution loop for',eval_budget,'iterations')
    while evalcount < eval_budget:
        pop_new_geno = np.zeros((mu, geno_len))
        #generate normal fitness
        total_fitness = sum(1/fitness)
        normal_fitness = np.divide(1/fitness, total_fitness)
        # generate the a new population using crossover and mutation
        for i in range(mu):
#selection part 1
            print('generating individual',i)
            p1 = np.random.choice(range(mu), p=normal_fitness)

            if np.random.randn() < pc:
#selection part 2 (optional)
                p2 = np.random.choice(range(mu), p=normal_fitness)
                while p1 == p2:
                    p2 = np.random.choice(range(mu), p=normal_fitness)
#crossover part (order 1 crossover chosen, because no repair needed by operation)
                #choose left point from [0..<highest index -1>]
                crossover_point_left = np.random.randint(0, high=geno_len-1)
                #choose right point from [<left point+1>..<highest index>]
                crossover_point_right = np.random.randint(crossover_point_left+1, high=geno_len)
                #copy same part
                for j in range(crossover_point_left,crossover_point_right+1):
                    pop_new_geno[i, j] = pop_geno[p1,j]
                tmp_index = 0
                tmp_checkarray = np.zeros(2*(geno_len-(crossover_point_right-1-crossover_point_left)))
                #copy parent2, last unused values
                for j in range(crossover_point_right+1, geno_len):
                    tmp_checkarray[tmp_index] = pop_geno[p2, j]
                    ++tmp_index
                #copy parent2, first unused values
                for j in range(0, crossover_point_left):
                    tmp_checkarray[tmp_index] = pop_geno[p2, j]
                    ++tmp_index
                #copy parent1, last unused values
                for j in range(crossover_point_right+1, geno_len):
                    tmp_checkarray[tmp_index] = pop_geno[p1, j]
                    ++tmp_index
                #copy parent1, first unused values
                for j in range(0, crossover_point_left):
                    tmp_checkarray[tmp_index] = pop_geno[p1, j]
                    ++tmp_index
                #copy all values in correct order in new geno, if unique afterwards
                cur_index = crossover_point_right+1
                while pop_new_geno[i,cur_index%geno_len] == 0:
                    for j in range(0, 2*(geno_len-(crossover_point_right-crossover_point_left))):
                        if not tmp_checkarray[j] in pop_new_geno[i, :]:
                            pop_new_geno[i,cur_index%geno_len] = tmp_checkarray[j]
                            ++cur_index
            else:
                # No crossover, copy the parent chromosome
                pop_new_geno[i] = p1
            if np.random.randn() < pm:
#swap mutation (chosen because no repair needed by operation + I am lazy)
                mutation_left = np.random.randint(0, high=geno_len-1)
                mutation_right= np.random.randint(mutation_left+1, high=geno_len)
                mutation_tmp_val = pop_new_geno[i, mutation_left]
                pop_new_geno[i, mutation_left] = pop_new_geno[i, mutation_right]
                pop_new_geno[i, mutation_right]= mutation_tmp_val

        # Replace old population by the newly generated population
        pop_geno = pop_new_geno

        for i in range(mu):
            fitness[i] = fitness_func(pop_geno[i, :])
            print('fitness',i,'/',mu,'determined')

        # optimal solution in each iteration
        index = np.argmin(fitness)
        x_opt_curr_gen = pop_geno[index, :]
        x_opt_pheno_curr_gen = pop_pheno[index, :]
        fopt_curr_gen = fitness[index]

        # keep track of the best solution ever found
        if fopt_curr_gen < fopt:
            fopt = fopt_curr_gen
            xopt = x_opt_curr_gen
            xopt_pheno = x_opt_pheno_curr_gen

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

            bar3.set_ydata(xopt_pheno)

            plt.draw()
        print('Iteration',i, '/',eval_budget,'complete')
    if return_stats:
        return xopt, fopt, hist_best_f
    else:
        return xopt, fopt
