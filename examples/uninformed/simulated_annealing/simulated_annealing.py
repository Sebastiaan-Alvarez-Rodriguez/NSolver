from nsolver.solver import Solver, evaluate

import configparser
from copy import copy
import numpy as np
import os


def get_solver():
    return SimulatedAnnealing


class SimulatedAnnealing(Solver):
    __version__ = 1.0

    '''Simulated annealing algorithm to solve magic N-cubes.'''
    def __init__(self, T=250000, alpha=0.9, pm=2, iter_length=100):
        self.T = T                     # Annealing temperature
        self.alpha = alpha             # temperature decaying parameter
        self.pm = pm                   # mutation rate
        self.iter_length = iter_length # number of evaluations per iteration


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
        if parser['NSolver']['solver'] != 'SimulatedAnnealing':
            raise ValueError(f'Config is made for another solver named "{parser["NSolver"]["solver"]}", expected "SimulatedAnnealing".')

        if float(parser['NSolver']['version']) != SimulatedAnnealing.__version__:
            raise ValueError(f'Expected to find version "{SimulatedAnnealing.__version__}", but found version "{parser["NSolver"]["version"]}"')
        default = parser['DEFAULT']
        return SimulatedAnnealing(T=int(default['T']), alpha=float(default['alpha']), pm=int(default['pm']), iter_length=int(default['iter_length']))


    @staticmethod
    def generate_random_answer(n, dim):
        '''Produces a valid, completely random answer.
        Note: An answer to the problem is not necessarily a solution to the problem.
              In fact, a random answer is unlikely to be a solution for the N-cube problem.
        Args:
            n (int): Axiomial dimension vector length (e.g., for a magic square of 3x3 fields (a 3-2D cube), n=3 and dim=2.
            dim (int): The amount of correlated dimensions (e.g., for a magic cube of x*x*x*x fields (a x-4D cube, n=x and dim=4.'''
        return np.random.permutation(np.arange(1, (n ** dim)+1))


    @staticmethod
    def mutate_answer(s, n, dim, pm, fitness, fitness_optimal):
        '''Mutates our current answer in hopes of finding a closer solution. Returns the permutated solution.
        The given mutation is a simple swap for a number of times, depending on the difference between the known optimum and given `s`, and `pm`.
        Args:
            s (list(int)): current answer to change. Note: Returns a new solution without modifying this parameter.
            n (int): Axiomial dimension vector length (e.g., for a magic square of 3x3 fields (a 3-2D cube), n=3 and dim=2.
            dim (int): The amount of correlated dimensions (e.g., for a magic cube of x*x*x*x fields (a x-4D cube, n=x and dim=4.
            pm (float): Permutation.
            fitness (double): The fitness value of our current solution `s`.
            fitness_optimal (double): The fitness value of the currently known most optimal solution.
        Returns:
            list(int): The permutated solution.'''

        length = n ** dim
        s_cpy = copy(s)

        # Do more mutations if we are far away from the optimal solution we found
        diff = max(fitness - fitness_optimal, 1)
        t_evals = int(np.ceil(diff * pm))
        
        # Mutate n times by swapping
        for i in range(1, t_evals):
            mut_left = np.random.randint(0, high=length-1)
            mut_right = np.random.randint(mut_left, high=length)
            s_cpy[mut_left], s_cpy[mut_right] = s_cpy[mut_right], s_cpy[mut_left]
        return s_cpy


    def execute(self, n, dim, evaluations, verbose):
        '''Execute this solver for the perfect cube problem, using given args.
        Args:
            n (int): Dimension size. e.g., for n=4 and dim=3, we expect a 4x4x4 cube.
            dim (int): Dimension of magic cube. E.g. for dim=2, must produce a magic square.
            evaluations (int): Maximum number of evaluations to perform.
        Returns:
            list(int): found solution.'''
        # if do_plot:
        #     plt.ion()
        #     fig = plt.figure()

        #     ax1 = plt.subplot(131)
        #     line1 = ax1.plot(hist_best_f[:evalcount])[0]
        #     ax1.set_title('minimal global error')
        #     ax1.set_ylabel('error')
        #     ax1.set_xlabel('evaluations')
        #     ax1.set_ylim([0, np.max(hist_best_f[:evalcount])])

        #     ax2 = plt.subplot(132)
        #     line2 = ax2.plot(np.arange(itercount), hist_temperature[:itercount])[0]
        #     ax2.set_title('temperature')
        #     ax2.set_ylabel('T')
        #     ax2.set_xlabel('iteration')
        #     ax2.set_ylim([0, T])

        #     ax3 = plt.subplot(133)
        #     bars3 = ax3.bar(np.arange(len(solution_optimal)), solution_optimal)
        #     ax3.set_title('best representation')
        #     ax3.set_ylabel('value')
        #     ax3.set_xlabel('representation index')

        #     plt.show(block=False)
        
        evalcount = 0
        itercount = 0

        # Statistics data
        hist_best_f = np.array([np.nan] * evaluations)
        num_iterations = int(np.ceil(evaluations / self.iter_length))
        hist_iter_f = np.array([np.nan] * num_iterations)
        hist_temperature = np.array([np.nan] * num_iterations)


        # Generate initial solution and evaluate
        solution_optimal = self.generate_random_answer(n, dim)
        fitness_optimal = evaluate(solution_optimal, n, dim=dim)  # evaluate the solution
        solution = copy(solution_optimal)
        fitness = fitness_optimal


        while evalcount < evaluations and fitness_optimal > 0.0: # We continue until we are out of budget or until we have found a solution
            hist_temperature[itercount] = self.T

            self.iter_length = min(self.iter_length, evaluations-evalcount)
            for _ in range(self.iter_length):

                solution_new = self.mutate_answer(solution, n, dim, self.pm, fitness, fitness_optimal)   # Generate a new solution by permutating the current solution
                fitness_new = evaluate(solution_new, n, dim=dim)   # evaluate the new solution

                if fitness_new < fitness or np.random.randn() < np.exp(-(fitness_new - fitness) / self.T):
                    # Our found mutation is closer to a solution than the current answer, or
                    # annealing formula mandates we pick this solution, even if it is a bit worse in terms of fitness.
                    solution = solution_new
                    fitness = fitness_new

                if fitness > 2 * fitness_optimal: 
                    # Reset to the optimal solution if we are too far away from found optimum.
                    solution = copy(solution_optimal)
                    fitness = fitness_optimal
                
                if fitness < fitness_optimal:
                    # Update the best solution found so far if our current solution is better.
                    fitness_optimal = fitness
                    solution_optimal = copy(solution)

                hist_best_f[evalcount] = fitness_optimal   # tracking the best fitness ever found

                # Generation best statistics
                hist_iter_f[itercount] = fitness
                
                # Plot statistics
                # if do_plot:
                #     line1.set_data(np.arange(evalcount), hist_best_f[:evalcount])
                #     ax1.set_xlim([0, evalcount])
                #     ax1.set_ylim([0, np.max(hist_best_f[:evalcount])])

                #     line2.set_data(np.arange(itercount), hist_temperature[:itercount])
                #     ax2.set_xlim([0, itercount])

                #     for bar, h in zip(bars3, solution_optimal):
                #         bar.set_height(h)

                #     plt.pause(0.00001)
                #     plt.draw()
                evalcount += 1   # Increase evaluation counter
            self.T = self.alpha * self.T

            if verbose:
                if verbose and evalcount % (evaluations/10) == 0:
                    print(f'{evalcount} (T={self.T:.02f}): current fitness: {fitness_optimal}')
            itercount += 1   # Increase iteration counter

        # if return_stats:
        #     return solution_optimal, fitness_optimal, hist_best_f
        # else:
        #     return solution_optimal, fitness_optimal
        return solution_optimal