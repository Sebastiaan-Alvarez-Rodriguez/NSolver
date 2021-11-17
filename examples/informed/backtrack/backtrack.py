from nsolver.solver import Solver, evaluate_correct, sum_row, sum_col, sum_diagonal, sum_vector, calc_magic_constant

import configparser
from copy import copy
import numpy as np
import os


def get_solver():
    return Backtrack


class Backtrack(Solver):
    _version_ = 1.0

    '''Backtracking algorithm to solve magic N-cubes.
    With backtracking, we walk across all possible states in a depth-first manner.
    When we find a valid solution, we stop searching.
    Many optimizations are possible with backtracking to reduce the search space.'''
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

    def execute(self, n, dim, evaluations, verbose):
        '''Execute this solver for the perfect cube problem, using given args.
        Args:
            n (int): Dimension size. e.g., for n=4 and dim=3, we expect a 4x4x4 cube.
            dim (int): Dimension of magic cube. E.g. for dim=2, must produce a magic square.
            evaluations (int): Maximum number of evaluations to perform.
        Returns:
            list(int): found solution.'''
        grid = np.zeros(n**dim, dtype=int)
        available_nums = np.array([True for x in range(n**dim)])
        if self._exec(grid, available_nums, 0, n, dim, n**dim, calc_magic_constant(n, dim)):
            return grid
        else:
            if verbose:
                print('Could not find a valid solution.')
            # return list(range(1, n**dim +1))
            return grid


    def _exec(self, grid, available_nums, idx, n, dim, max_len, magic_constant):
        '''Recursive function performing the actual backtracking.
        Args:
            grid (np.array(int)): Array representing the current solution state.
            available_nums (np.array(bool)): Array representing the available numbers to use.
            idx (int): Current index to evaluate. Recursive iterations will validate the next indices.
            dim (int): Dimension of the problem.
            max_len (int): length of the grid.
            magic_constant (int): magic cubes must have rows, columns and diagonals with a sum equal to this value.
        Returns:
            bool: `True` if we found a solution. `False` otherwise.'''
        if idx == max_len: # When our grid is filled, we evaluate the solution. Return whether it is correct.
            return evaluate_correct(grid, dim=dim)

        if idx % n == 0 and idx > 0: # We completed a row. Validate whether the row contains generics.
            if sum_row(grid, n, dim, idx//n-1) != magic_constant:
                return False
        if idx > n*(n-1):
            if sum_col(grid, n, dim, idx-n*(n-1)-1) != magic_constant:
                return False
            # print(f'({idx}) Col idx={idx-n*(n-1)-1}, {grid} ---> ({sum_col(grid, n, dim, idx-n*(n-1)-1)})')

        # Sanity check: Verifies whether all available items are actually available.
        # for i, x in enumerate(list(available_nums)):
        #     num_ = i+1
        #     if x and num_ in grid:
        #         raise ValueError(f'(idx={idx}, num there={grid[idx]}) Found number={num_} in grid ({grid}) while labeled "available" in state array: {available_nums}')
        for num in list(_available_get_unused(available_nums)):
            grid[idx] = num
            _available_set_available(available_nums, num, False)
            if self._exec(grid, available_nums, idx+1, n, dim, max_len, magic_constant):
                return True
            _available_set_available(available_nums, num) # TODO: What happens when updating array when iterating over it?
        return False

# A valid solution:
#
# 08 03 04
# 01 05 09
# 06 07 02


def _available_get_unused_next(available_nums):
    return next(_available_get_unused(available_nums))

def _available_get_unused(available_nums):
    '''fetches the next unused number.
    Returns:
        int: next unused number. 
    Raises:
        StopIteration: If all numbers are already used'''
    return (idx+1 for idx, x in enumerate(available_nums) if x)

def _available_set_available(available_nums, number, value=True):
    available_nums[number-1] = value
