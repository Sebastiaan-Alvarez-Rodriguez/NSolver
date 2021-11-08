import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from importlib import import_module
from mpl_toolkits import mplot3d


def get_solver():
    return GeneticAlgorithm

class GeneticAlgorithm(Solver):

    def __init__(self):
        # endogenous parameters setting
        mu = n*dim        # population size
        pc = 0.9          # crossover rate
        pm = 0.3          # mutation rate

        # internal counter variable
        evalcount = 0     # count function evaluations
        gencount = 0      # count generation/iterations


    def execute(self, n, dim, evaluations):
        '''Execute this solver for the perfect cube problem, using given args.
        Args:
            n (list(int)): List of numbers to form a magic cube. The first n entries form row 0, the next n entries row 1, etc.
            dim (int): Dimension of magic cube. E.g. for dim=2, must produce a magic square.
            evaluations (int): Maximum number of evaluations to perform.
        Returns:
            list(int): found solution.'''

        # ----------------- general setting of variables ----------------------
        geno_len = n**dim   # Representation: Array of integers with values [1, n^dim]

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
        # if do_plot:
        #     # plt.ion()
        #     ax1 = plt.subplot(131)
        #     line1 = ax1.plot(hist_best_f[:evalcount])[0]
        #     ax1.set_title('min. global error')
        #     ax1.set_ylabel('error')
        #     ax1.set_xlabel('evaluations')
        #     ax1.set_ylim([0, np.max(hist_best_f)])

        #     ax2 = plt.subplot(132)
        #     line2 = ax2.plot(hist_gen_f[:gencount])[0]
        #     ax2.set_title('min. error in cur gen')
        #     ax2.set_ylabel('error')
        #     ax2.set_xlabel('generation')
        #     ax2.set_ylim([0, np.argmax(fitness)])

        #     ax3 = plt.subplot(133)
        #     bars3 = ax3.bar(np.arange(geno_len), xopt)
        #     ax3.set_title('best chromosome')
        #     ax3.set_ylabel('value')
        #     ax3.set_xlabel('genotype index')

        #     plt.show(block=False)

        #     if dim == 3:
        #     X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))

        #     fig = plt.figure(figsize=(9,7))
        #     ax0 = fig.gca(projection='3d')

        #     fig.suptitle("Best cube found using ga")
        #     ax0.xaxis.set_ticks([])
        #     ax0.yaxis.set_ticks([])
        #     ax0.zaxis.set_ticks([])
        #     ax0.set_ylim(bottom=0, top=n)
        #     ax0.set_xlim(left=0, right=n)
        #     ax0.set_zlim(bottom=0, top=n)
        #     ax0.view_init(elev=10, azim=-85)
        #     fig.tight_layout(pad=5, h_pad=0, w_pad=0, rect=[-.25,-.25,1.25,1.25])
        #     cube = xopt.reshape((n, n, n))
        #     X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
        #     text = [... for i in range(n**dim)]
        #     for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten()):
        #         text[z*n*n+y*n+x] = ax0.text(x+.5, y+.5, z+.5, s=str(int(cube[y,x,z])), color=plt.cm.winter(y/n), fontsize=7, horizontalalignment='center', verticalalignment='center')
        # elif dim == 2:
        #     fig = plt.figure(figsize=(9,7))
        #     fig.show()
        #     fig.canvas.draw()

        #     plt.title("Best square found using ga")
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.ylim(top=n, bottom=0)
        #     plt.xlim(left=0, right=n)
        #     plt.tight_layout()
        #     for x in np.arange(n):
        #         plt.axhline(x+1, color='black', linewidth=1)
        #         plt.axvline(x+1, color='black', linewidth=1)
        #     square = xopt.reshape((n, n))
        #     X, Y = np.meshgrid(np.arange(n), np.arange(n))
        #     text = ['' for i in range(n**dim)]
        #     for x, y in zip(X.flatten(), Y.flatten()):
        #         text[y*n+x] = plt.text(x+.5, n-y-.5, s=str(int(square[y,x])), color=plt.cm.winter(square[y,x]/n**2), horizontalalignment='center', verticalalignment='center')

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

                if np.random.randn() < pc: #selection part 2 (optional)
                    p2 = np.random.choice(range(mu), p=normal_fitness, replace=False) 
                    # order 1 crossover part

                    crossover_point_left = np.random.randint(0, high=geno_len-1) # choose left point from [0..<highest index -1>]
                    crossover_point_right = np.random.randint(crossover_point_left+1, high=geno_len) # choose right point from [<left point+1>..<highest index>]
                    for j in range(crossover_point_left,crossover_point_right+1):
                        pop_new_geno[i, j] = pop_geno[p1,j]
                    tmp_index = 0
                    tmp_checkarray = np.zeros(2*(geno_len-(crossover_point_right-1-crossover_point_left)))
                    for idx, j in enumerate(range(crossover_point_right, geno_len)): #copy parent2, last unused values
                        tmp_checkarray[idx] = pop_geno[p2, j]
                    tmp_index += geno_len-crossover_point_right

                    for idx, j in enumerate(range(0, crossover_point_left)): #copy parent2, first unused values
                        tmp_checkarray[tmp_index+idx] = pop_geno[p2, j]
                    tmp_index += crossover_point_left

                    for idx, j in enumerate(range(crossover_point_right, geno_len)): #copy parent1, last unused values
                        tmp_checkarray[tmp_index+idx] = pop_geno[p1, j]
                    tmp_index += geno_len-crossover_point_right

                    for idx, j in enumerate(range(0, crossover_point_left)): #copy parent1, first unused values
                        tmp_checkarray[tmp_index] = pop_geno[p1, j]
                    tmp_index += crossover_point_left

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

            # keep track of the best solution found
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
                    fig.suptitle(f"Best cube found using ga ({evalcount}/{eval_budget}), fitness={fopt}")
                    cube = x_opt_curr_gen.reshape((n, n, n))
                    X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
                    for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten()):
                        text[z*n*n+y*n+x].remove()
                        text[z*n*n+y*n+x] = ax0.text(x+.5, y+.5, z+.5, s=str(int(cube[y,x,z])), color=plt.cm.winter(y/n), fontsize=7, horizontalalignment='center', verticalalignment='center')

                elif dim == 2:
                    plt.title(f'Best square found using ga ({evalcount}/{eval_budget}), fitness={round(fopt, 2)}')
                    square = x_opt_curr_gen.reshape((n, n))
                    X, Y = np.meshgrid(np.arange(n), np.arange(n))

                    for x, y in zip(X.flatten(), Y.flatten()):
                        text[y*n+x].remove()
                        text[y*n+x] = plt.text(x+.5, n-y-.5, s=str(int(square[y,x])), color=plt.cm.winter(square[y,x]/n**2), horizontalalignment='center', verticalalignment='center')
                if plt.fignum_exists(fig.number):
                    fig.canvas.draw()
                else:
                    exit()

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
        plt.clf()
        plt.close('all')
        if return_stats:
            return xopt, fopt, hist_best_f
        else:
            return xopt, fopt