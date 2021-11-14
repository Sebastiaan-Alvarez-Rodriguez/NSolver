from nsolver.solver import Solver, evaluate

import configparser
from copy import copy
import numpy as np
import os


def get_solver():
    return Backtrack


class Backtrack(Solver):
    __version__ = 1.0

    '''Backtracking algorithm to solve magic N-cubes.'''
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
        if parser['NSolver']['solver'] != 'Backtracking':
            raise ValueError(f'Config is made for another solver named "{parser["NSolver"]["solver"]}", expected "Backtracking".')

        if float(parser['NSolver']['version']) != Backtracking.__version__:
            raise ValueError(f'Expected to find version "{Backtracking.__version__}", but found version "{parser["NSolver"]["version"]}"')
        default = parser['DEFAULT']
        return Backtracking(T=int(default['T']), alpha=float(default['alpha']), pm=int(default['pm']), iter_length=int(default['iter_length']))


    @staticmethod
    def generate_random_answer(n, dim):
        '''Produces a valid, completely random answer.
        Note: An answer to the problem is not necessarily a solution to the problem.
              In fact, a random answer is unlikely to be a solution for the N-cube problem.
        Args:
            n (int): Axiomial dimension vector length (e.g., for a magic square of 3x3 fields (a 3-2D cube), n=3 and dim=2.
            dim (int): The amount of correlated dimensions (e.g., for a magic cube of x*x*x*x fields (a x-4D cube, n=x and dim=4.'''
        return np.random.permutation(np.arange(1, (n ** dim)+1))

    def sum_row(grid, n, dim, row_idx):
        return sum(grid[row_idx*n:row_idx*(n+1)])

    def sum_col(grid, n, dim, col_idx):
        return np.sum(grid[col_idx:col_idx+n**2::n])

    def sum_diagonal(grid, n, dim, col_idx):
        pass
        '''
np.array(a).reshape(4, 4, 4)
array([[[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.]],

       [[16., 17., 18., 19.],
        [20., 21., 22., 23.],
        [24., 25., 26., 27.],
        [28., 29., 30., 31.]],

       [[32., 33., 34., 35.],
        [36., 37., 38., 39.],
        [40., 41., 42., 43.],
        [44., 45., 46., 47.]],

       [[48., 49., 50., 51.],
        [52., 53., 54., 55.],
        [56., 57., 58., 59.],
        [60., 61., 62., 63.]]])

        '''

    def execute(self, n, dim, evaluations, verbose):
        '''Perform backtracking for the magic N-cube problem, using given args.
        Args:
            n (list(int)): List of numbers to form a magic cube. The first n entries form row 0, the next n entries row 1, etc.
            dim (int): Dimension of magic cube. E.g. for dim=2, must produce a magic square.
            evaluations (int): Maximum number of evaluations to perform.
            verbose (bool): If set, print more output.
        Returns:
            list(int): found solution.'''
        
        grid = np.zeros(n**dim)


        # Generate initial solution and evaluate
        solution_optimal = self.generate_random_answer(n, dim)
        fitness_optimal = evaluate(solution_optimal, dim=dim)  # evaluate the solution
        solution = copy(solution_optimal)
        fitness = fitness_optimal


        while evalcount < evaluations and fitness_optimal > 0.0: # We continue until we are out of budget or until we have found a solution
            hist_temperature[itercount] = self.T

            self.iter_length = min(self.iter_length, evaluations-evalcount)
            for _ in range(self.iter_length):

                solution_new = self.mutate_answer(solution, n, dim, self.pm, fitness, fitness_optimal)   # Generate a new solution by permutating the current solution
                fitness_new = evaluate(solution_new, dim=dim)   # evaluate the new solution

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