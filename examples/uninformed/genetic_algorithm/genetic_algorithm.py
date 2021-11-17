import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

from nsolver.solver import Solver, evaluate

import configparser
from copy import copy
import numpy as np
import os

def get_solver():
    return GeneticAlgorithm

class GeneticAlgorithm(Solver):
    __version__ = 1.0

    '''Simulated annealing algorithm to solve magic N-cubes.'''
    def __init__(self, pc=0.7, pm=0.7, fitness_weight=4.0):
        # endogenous parameters setting
        self.pc = pc          # crossover rate
        self.pm = pm          # mutation rate
        self.fitness_weight = fitness_weight # impact of fitness on chance of being chosen


    @staticmethod
    def from_config(path):
        '''Builds this Solver using given configuration file.
        Args:
            path (str or Path): path to configuration file.
        Returns:
            Solver: Solver implementation created from the configuration file.'''
        parser = configparser.ConfigParser()
        parser.optionxform=str
        parser.read(path)
        if parser['NSolver']['solver'] != 'GeneticAlgorithm':
            raise ValueError(f'Config is made for another solver named "{parser["NSolver"]["solver"]}", expected "GeneticAlgorithm".')

        if float(parser['NSolver']['version']) != GeneticAlgorithm.__version__:
            raise ValueError(f'Expected to find version "{GeneticAlgorithm.__version__}", but found version "{parser["NSolver"]["version"]}"')
        default = parser['DEFAULT']
        return GeneticAlgorithm(pc=float(default['pc']), pm=float(default['pm']), fitness_weight=float(default['fitness_weight']))



    def execute(self, n, dim, evaluations, verbose):
        '''Execute this solver for the perfect cube problem, using given args.
        Args:
            n (int): Dimension size. e.g., for n=4 and dim=3, we expect a 4x4x4 cube.
            dim (int): Dimension of magic cube. E.g. for dim=2, must produce a magic square.
            evaluations (int): Maximum number of evaluations to perform.
        Returns:
            list(int): found solution.'''


        # ----------------- general setting of variables ----------------------
        geno_len = n**dim   # Representation: Array of integers with values [1, n^dim]

        # endogenous parameters setting
        mu = n*dim        # population size

        # internal counter variable
        evalcount = 0     # count function evaluations
        gencount = 0      # count generation/iterations


        # historical information
        hist_best_f = np.zeros(evaluations)
        hist_gen_f = np.zeros(int(np.ceil(evaluations/mu)))


        # ------------------- population initialization -----------------------
        # row vector representation is used throughout this code
        pop_fitness = np.zeros(mu)
        # population
        pop_geno = np.zeros((mu, geno_len), dtype=int)

        # Initialize all members of population a random value [1, n^dim]) and evaluates
        for i in range(mu):
            pop_geno[i, :] = np.random.permutation(range(1, geno_len+1))
            pop_fitness[i] = evaluate(pop_geno[i, :], n, dim=dim)
        index = np.argmin(pop_fitness)

        # Generate initial solution and evaluate

        solution_optimal = pop_geno[index, :]
        fitness_optimal = pop_fitness[index]

        hist_best_f[evalcount:evalcount+mu] = fitness_optimal
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
        #     bars3 = ax3.bar(np.arange(geno_len), solution_optimal)
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
        #     cube = solution_optimal.reshape((n, n, n))
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
        #     square = solution_optimal.reshape((n, n))
        #     X, Y = np.meshgrid(np.arange(n), np.arange(n))
        #     text = ['' for i in range(n**dim)]
        #     for x, y in zip(X.flatten(), Y.flatten()):
        #         text[y*n+x] = plt.text(x+.5, n-y-.5, s=str(int(square[y,x])), color=plt.cm.winter(square[y,x]/n**2), horizontalalignment='center', verticalalignment='center')

        # ----------------------- Evolution loop ------------------------------
        # https://blackboard.leidenuniv.nl/bbcswebdav/pid-4458530-dt-content-rid-5832938_1/courses/4032NACO6-1819FWN/3b%20-%20Evolutionary_Algorithms.pdf

        while evalcount < evaluations and fitness_optimal > 0.0:
            #generate normal fitness
            total_fitness = sum(1/(pop_fitness**self.fitness_weight))
            normal_fitness = np.divide(1/(pop_fitness**self.fitness_weight), total_fitness)
            # generate the a new population using crossover and mutation
            for i in range(mu):
                p1 = np.random.choice(range(mu), p=normal_fitness, replace=False)

                if np.random.randn() < self.pc: # order 1 crossover
                    # pop1: A B C  --> A E C
                    # pop2: D E F  --> D B F
                    p2 = np.random.choice(range(mu), p=normal_fitness, replace=False) 

                    crossover_point_left = np.random.randint(0, high=geno_len-1) # choose left point from [0..<highest index -1>]
                    crossover_point_right = np.random.randint(crossover_point_left+1, high=geno_len) # choose right point from [<left point+1>..<highest index>]

                    for x in range(crossover_point_left, crossover_point_right): # In-place crossover (is there std::swap maybe?)
                        backup = pop_geno[p1][x]
                        pop_geno[p1][x] = pop_geno[p2][x]
                        pop_geno[p2][x] = backup

                    # Repair broken genotypes: After crossover, there are potentially double and missing items in both pops.
                    for arr in [pop_geno[p1], pop_geno[p2]]:
                        seen_it = [False for x in range(geno_len)]
                        for idx, x in enumerate(arr):
                            if not seen_it[x-1]:
                                seen_it[x-1] = True
                            else: # We got a double occurance
                                new_val = next(idx2+1 for idx2, x2 in enumerate(seen_it) if not x2)
                                arr[idx] = new_val
                                seen_it[new_val-1] = True

                if np.random.randn() < self.pm: # swap mutation (no repair needed by operation)
                    mutation_left = np.random.randint(0, high=geno_len-1)
                    mutation_right= np.random.randint(mutation_left+1, high=geno_len)
                    pop_geno[i, mutation_left], pop_geno[i, mutation_right] = pop_geno[i, mutation_right], pop_geno[i, mutation_left]


            for i in range(mu):
                pop_fitness[i] = evaluate(pop_geno[i, :], n, dim=dim)

            # optimal solution in each iteration
            index = np.argmin(pop_fitness)
            solution_optimal_curr_gen = pop_geno[index, :]
            fitness_optimal_curr_gen = pop_fitness[index]

            # keep track of the best solution found
            if fitness_optimal_curr_gen < fitness_optimal:
                fitness_optimal = fitness_optimal_curr_gen
                solution_optimal = solution_optimal_curr_gen

            # record historical information
            hist_best_f[evalcount:evalcount+mu] = fitness_optimal
            hist_gen_f[gencount] = fitness_optimal_curr_gen

            # internal counters increment
            gencount += 1
            evalcount += mu

            if verbose and evalcount % (evaluations/10) == 0:
                print(f'{evalcount}: current fitness: {fitness_optimal}')

            # Plot statistics
            # if do_plot:
            #     if dim == 3:
            #         fig.suptitle(f"Best cube found using ga ({evalcount}/{evaluations}), fitness={fitness_optimal}")
            #         cube = solution_optimal_curr_gen.reshape((n, n, n))
            #         X, Y, Z = np.meshgrid(np.arange(n), np.arange(n), np.arange(n))
            #         for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten()):
            #             text[z*n*n+y*n+x].remove()
            #             text[z*n*n+y*n+x] = ax0.text(x+.5, y+.5, z+.5, s=str(int(cube[y,x,z])), color=plt.cm.winter(y/n), fontsize=7, horizontalalignment='center', verticalalignment='center')
            #     elif dim == 2:
            #         plt.title(f'Best square found using ga ({evalcount}/{evaluations}), fitness={round(fitness_optimal, 2)}')
            #         square = solution_optimal_curr_gen.reshape((n, n))
            #         X, Y = np.meshgrid(np.arange(n), np.arange(n))
            #         for x, y in zip(X.flatten(), Y.flatten()):
            #             text[y*n+x].remove()
            #             text[y*n+x] = plt.text(x+.5, n-y-.5, s=str(int(square[y,x])), color=plt.cm.winter(square[y,x]/n**2), horizontalalignment='center', verticalalignment='center')
            #     if plt.fignum_exists(fig.number):
            #         fig.canvas.draw()
            #     else:
            #         exit()

            #     line1.set_data(np.arange(evalcount), hist_best_f[:evalcount])
            #     ax1.set_xlim([0, evalcount])
            #     ax1.set_ylim([0, np.max(hist_best_f)])

            #     line2.set_data(np.arange(gencount), hist_gen_f[:gencount])
            #     ax2.set_xlim([0, gencount])
            #     ax2.set_ylim([0, np.max(hist_gen_f)])

            #     for bar, h in zip(bars3, solution_optimal):
            #         bar.set_height(h)

            #     plt.pause(0.00001)
            #     plt.draw()
        # plt.clf()
        # plt.close('all')
        return solution_optimal