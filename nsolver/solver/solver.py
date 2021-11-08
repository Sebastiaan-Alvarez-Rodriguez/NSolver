

def get_solver():
    '''Returns the implemented solver class.'''
    return solver

class Solver(object):
    def __init__(self):
        pass


    @staticmethod
    def from_config(path):
        '''Builds this Solver using given configuration file.
        Args:
            path (str or Path): path to configuration file.
        Returns:
            Solver: Solver implementation created from the configuration file.'''
        raise NotImplementedError('This solver does not support loading from a configuration file.')


    def execute(self, n, dim, evaluations):
        '''Execute this solver for the perfect cube problem, using given args.
        Args:
            n (list(int)): List of numbers to form a magic cube. The first n entries form row 0, the next n entries row 1, etc.
            dim (int): Dimension of magic cube. E.g. for dim=2, must produce a magic square.
            evaluations (int): Maximum number of evaluations to perform.
        Returns:
            list(int): found solution.'''
        raise NotImplementedError