from nsolver.solver import Solver, evaluate

import configparser
from copy import copy
import numpy as np
import os


def get_solver():
    return Random


class Random(Solver):
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
        if parser['NSolver']['solver'] != 'Random':
            raise ValueError(f'Config is made for another solver named "{parser["NSolver"]["solver"]}", expected "Random".')

        if float(parser['NSolver']['version']) != Random.__version__:
            raise ValueError(f'Expected to find version "{Random.__version__}", but found version "{parser["NSolver"]["version"]}"')
        default = parser['DEFAULT']
        return Random(T=int(default['T']), alpha=float(default['alpha']), pm=int(default['pm']), iter_length=int(default['iter_length']))


    @staticmethod
    def generate_random_answer(n, dim):
        '''Produces a valid, completely random answer.
        Note: An answer to the problem is not necessarily a solution to the problem.
              In fact, a random answer is unlikely to be a solution for the N-cube problem.
        Args:
            n (int): Axiomial dimension vector length (e.g., for a magic square of 3x3 fields (a 3-2D cube), n=3 and dim=2.
            dim (int): The amount of correlated dimensions (e.g., for a magic cube of x*x*x*x fields (a x-4D cube, n=x and dim=4.'''
        return np.random.permutation(np.arange(1, (n ** dim)+1))



    def execute(self, n, dim, evaluations, verbose):
        '''Execute this solver for the perfect cube problem, using given args.
        Args:
            n (list(int)): List of numbers to form a magic cube. The first n entries form row 0, the next n entries row 1, etc.
            dim (int): Dimension of magic cube. E.g. for dim=2, must produce a magic square.
            evaluations (int): Maximum number of evaluations to perform.
        Returns:
            list(int): found solution.'''
        solution_optimal = None
        fitness_optimal = np.inf
        for evalcount, x in enumerate(range(evaluations)):
            cur_answer = Random.generate_random_answer(n, dim)
            cur_fitness = evaluate(cur_answer, dim=dim)
            if cur_fitness < fitness_optimal:
                solution_optimal = cur_answer
                fitness_optimal = cur_fitness

            if verbose and evalcount % (evaluations/10) == 0:
                print(f'{evalcount}: current fitness: {fitness_optimal}')

        return solution_optimal