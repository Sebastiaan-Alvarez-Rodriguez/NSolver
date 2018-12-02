import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from importlib import import_module
from mpl_toolkits import mplot3d

def ga_skeleton(dim, eval_budget, fitness_func, do_plot=False, return_stats=False):
    # ----------------- general setting of variables ----------------------

    if dim == 2:
        n = 12
    elif dim == 3:
        n = 7
    else:
        raise ValueError('Invalid number of dimensions, use 2 or 3')

    #Representation: Array of integers with values [1, n^dim]
    geno_len = n**dim 


    # endogenous parameters setting
    mu = n*dim        # population size
    pc = 0.9          # crossover rate
    pm = 0.3          # mutation rate

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

    # Initialize all members of population a random value [1, n^dim]) and evaluates
    for i in range(mu):
        pop_geno[i, :] = np.random.permutation(np.arange(1, geno_len+1))
        fitness[i] = fitness_func(pop_geno[i, :])
    index = np.argmin(fitness)
    fopt = fitness[index]
    xopt = pop_geno[index, :]

    hist_best_f[evalcount:evalcount+mu] = fopt
    evalcount += mu

    # ----------------------- config drawing ------------------------------
    if do_plot:
        plt.ion()
    if dim == 3:
        semi_perfect = False #TODO: get whether this is semi perfect or not
        X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
        semi = 'semi-perfect ' if semi_perfect else ''

        fig = plt.figure(figsize=(9,7))
        ax = fig.gca(projection='3d')

        plt.title("Best cube found using ga")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.zaxis.set_ticks([])
        ax.set_ylim(bottom=0, top=n)
        ax.set_xlim(left=0, right=n)
        ax.set_zlim(bottom=0, top=n)
        ax.view_init(elev=10, azim=-85)
        plt.tight_layout(pad=5, h_pad=0, w_pad=0, rect=[-.25,-.25,1.25,1.25])
        cube = xopt.reshape((n, n, n))
        X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
        text = [... for i in range(n**dim)]
        for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten()):
            text[z*n*n+y*n+x] = ax.text(x+.5, y+.5, z+.5, s=str(int(cube[y,x,z])), color=plt.cm.winter(y/n), fontsize=7, horizontalalignment='center', verticalalignment='center')
    elif dim == 2:
        fig = plt.figure(figsize=(9,7))
        fig.show()
        fig.canvas.draw()
        plt.title("Best square found using ga")
        plt.xticks([])
        plt.yticks([])
        plt.ylim(top=n, bottom=0)
        plt.xlim(left=0, right=n)
        plt.tight_layout()
        for x in np.arange(n):
            plt.axhline(x+1, color='black', linewidth=1)
            plt.axvline(x+1, color='black', linewidth=1)
        square = xopt.reshape((n, n))
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        text = ['' for i in range(n**dim)]
        for x, y in zip(X.flatten(), Y.flatten()):
            text[y*n+x] = plt.text(x+.5, n-y-.5, s=str(int(square[y,x])), color=plt.cm.winter(square[y,x]/n**2), horizontalalignment='center', verticalalignment='center')

    # ----------------------- Evolution loop ------------------------------
#https://blackboard.leidenuniv.nl/bbcswebdav/pid-4458530-dt-content-rid-5832938_1/courses/4032NACO6-1819FWN/3b%20-%20Evolutionary_Algorithms.pdf
    while evalcount < eval_budget:
        pop_new_geno = np.zeros((mu, geno_len))
        #generate normal fitness
        total_fitness = sum(1/(fitness**30))
        normal_fitness = np.divide(1/(fitness**30), total_fitness)
        # generate the a new population using crossover and mutation
        for i in range(mu):
#selection part 1
            p1 = np.random.choice(range(mu), p=normal_fitness, replace=False)

            if np.random.randn() < pc:
#selection part 2 (optional)
                p2 = np.random.choice(range(mu), p=normal_fitness, replace=False)
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
                for j in range(crossover_point_right, geno_len):
                    tmp_checkarray[tmp_index] = pop_geno[p2, j]
                    tmp_index += 1
                #copy parent2, first unused values
                for j in range(0, crossover_point_left):
                    tmp_checkarray[tmp_index] = pop_geno[p2, j]
                    tmp_index += 1
                #copy parent1, last unused values
                for j in range(crossover_point_right, geno_len):
                    tmp_checkarray[tmp_index] = pop_geno[p1, j]
                    tmp_index += 1
                #copy parent1, first unused values
                for j in range(0, crossover_point_left):
                    tmp_checkarray[tmp_index] = pop_geno[p1, j]
                    tmp_index += 1
                #copy all values in correct order in new geno, if unique afterwards
                cur_index = crossover_point_right+1
                while pop_new_geno[i,cur_index%geno_len] == 0:
                    for j in range(0, 2*(geno_len-(crossover_point_right-crossover_point_left))):
                        if not tmp_checkarray[j] in pop_new_geno[i, :]:
                            pop_new_geno[i,cur_index%geno_len] = tmp_checkarray[j]
                            cur_index += 1
            else:
                # No crossover, copy parent chromosome
                pop_new_geno[i] = pop_geno[p1]
            if np.random.randn() < (pm):
#swap mutation (chosen because no repair needed by operation + I am lazy)
                mutation_left = np.random.randint(0, high=geno_len-1)
                mutation_right= np.random.randint(mutation_left+1, high=geno_len)
                pop_new_geno[i, mutation_left], pop_new_geno[i, mutation_right] = pop_new_geno[i, mutation_right], pop_new_geno[i, mutation_left]

        # Replace old population by the newly generated population
        pop_geno = pop_new_geno

        for i in range(mu):
            fitness[i] = fitness_func(pop_geno[i, :])

        # optimal solution in each iteration
        index = np.argmin(fitness)
        x_opt_curr_gen = pop_geno[index, :]
        fopt_curr_gen = fitness[index]

        # keep track of the best solution ever found
        if fopt_curr_gen < fopt:
            fopt = fopt_curr_gen
            xopt = x_opt_curr_gen

        # record historical information
        hist_best_f[evalcount:evalcount+mu] = fopt
        hist_gen_f[gencount] = fopt_curr_gen

        # internal counters increment
        gencount += 1
        evalcount += mu

        # Plot statistics
        if do_plot:
            if dim == 3:
                plt.title(f"Best cube found using ga ({evalcount}/{eval_budget}), fitness={fopt}")
                cube = x_opt_curr_gen.reshape((n, n, n))
                X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
                for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten()):
                    text[z*n*n+y*n+x].remove()
                    text[z*n*n+y*n+x] = ax.text(x+.5, y+.5, z+.5, s=str(int(cube[y,x,z])), color=plt.cm.winter(y/n), fontsize=7, horizontalalignment='center', verticalalignment='center')
                fig.canvas.draw()

            elif dim == 2:
                plt.title(f'Best square found using ga ({evalcount}/{eval_budget}), fitness={round(fopt, 2)}')
                square = x_opt_curr_gen.reshape((n, n))
                X, Y = np.meshgrid(np.arange(n), np.arange(n))

                for x, y in zip(X.flatten(), Y.flatten()):
                    text[y*n+x].remove()
                    text[y*n+x] = plt.text(x+.5, n-y-.5, s=str(int(square[y,x])), color=plt.cm.winter(square[y,x]/n**2), horizontalalignment='center', verticalalignment='center')
                fig.canvas.draw()
    plt.clf()
    plt.close()
    if return_stats:
        return xopt, fopt, hist_best_f
    else:
        return xopt, fopt