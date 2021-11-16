from nsolver.solver import Solver, evaluate_correct

import configparser
from copy import copy
import numpy as np
import os


def get_solver():
    return Backtrack


class Backtrack(Solver):
    _version_ = 1.0

    '''Backtracking algorithm to solve magic N-cubes.'''
    def _init_(self, T=250000, alpha=0.9, pm=2, iter_length=100):
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

        if float(parser['NSolver']['version']) != Backtracking._version_:
            raise ValueError(f'Expected to find version "{Backtracking._version_}", but found version "{parser["NSolver"]["version"]}"')
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

    def sum_vector(grid, n, dim, row_idx, dim_idx):
        if dim_idx == 0: # dim_idx = 0 --> x-axis (rows). Rows are sequential numbers in the array, starting on any idx % n == 0.
            return np.sum(grid[row_idx*n:row_idx*(n+1)])
        if dim_idx == 1: # dim_idx = 1 --> y-axis (cols). Cols are numbers in the array with n-length jumps between them. 
            return np.sum(grid[row_idx:row_idx+n**2::n])
        return np.sum(grid[row_idx:row_idx+n**(dim_idx+1)::n**dim_idx])

    def sum_row(grid, n, dim, row_idx):
        return sum_vector(grid, n, dim, row_idx, 0)

    def sum_col(grid, n, dim, col_idx):
        return sum_vector(grid, n, dim, row_idx, 1)

    def sum_diagonal(grid, n, dim, col_idx):
        pass

    def execute(self, n, dim, evaluations, verbose):
        '''Perform backtracking for the magic N-cube problem, using given args.
        Args:
            n (list(int)): List of numbers to form a magic cube. The first n entries form row 0, the next n entries row 1, etc.
            dim (int): Dimension of magic cube. E.g. for dim=2, must produce a magic square.
            evaluations (int): Maximum number of evaluations to perform.
            verbose (bool): If set, print more output.
        Returns:
            list(int): found solution.'''
        grid = np.zeros(n**dim, dtype=int)
        available_nums = np.array([True for x in range(n**dim)])
        self._exec(grid, available_nums, 0, n**dim, dim)
        return grid


    def _exec(self, grid, available_nums, idx, max_len, dim):
        '''Recursive function performing the actual backtracking.
        Args:
            grid (np.array(int)): Array representing the current solution state.
            available_nums (np.array(bool)): Array representing the available numbers to use.
            idx (int): Current index to evaluate. Recursive iterations will validate the next indices.
            max_len (int): length of the grid.
            dim (int): Dimension of the problem.
        Returns:
            bool: `True` if we found a solution. `False` otherwise.'''
        if idx == max_len: # If our grid is filled, we evaluate the correctness. If it is a correct answer, return it.
            return evaluate_correct(grid, dim=dim)

        for num in _available_get_unused(available_nums):
            grid[idx] = num
            _available_set_available(available_nums, num, False)
            if self._exec(grid, available_nums, idx+1, max_len, dim):
                return True
            _available_set_available(available_nums, num) # TODO: What happens when updating array when iterating over it?
        return False


def _available_get_unused_next(available_nums):
    return next(_available_get_unused(available_nums))

def _available_get_unused(available_nums):
    '''fetches the next unused number.
    Returns:
        int: next unused number. 
    Raises:
        StopIteration: If all numbers are already used'''
    return (idx+1 for x in enumerate(available_nums) if x)

def _available_set_available(self, available_nums, number, value=True):
    available_nums[number-1] = value
