from nsolver.solver import Solver, evaluate_correct, sum_available, sum_closest_available, sum_vector, calc_magic_constant

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

        for d in range(dim):
            closest_row_idx, available = sum_closest_available(grid, n, dim, d, idx)
            if available:
                if sum_vector(grid, n, dim, closest_row_idx, d) != magic_constant:
                    return False

        if idx == max_len: # When our grid is filled, we evaluate the solution. Return whether it is correct.
            return evaluate_correct(grid, dim=dim)

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
