from nsolver import get_solver, is_solution, evaluate
from nsolver.utils.printer import *
from .util import print_solution
from .run import execute

import inspect
import time


'''Simple search system to find good parameters for a solver.'''


__name__ = 'search'
__help__ = 'Search for "good" parameters for a solver.'


def build_cli(parser):
    '''Adds run commandline interface (cli) parser and arguments.
    Args:
        parser (argparse.Parser): Base parser to extend.'''
    parser.add_argument('path', metavar='path', type=str, help='Path to solver implementation.')
    parser.add_argument('--evaluations', metavar='amount', type=int, default=10000, help='Amount of evaluations per parser (default=10,000). A higher number takes longer.')
    parser.add_argument('--filters', metavar='filter[:filter]...]', type=str, default=None, help='If set, only executes solvers with names matching given regex. To supply multiple regexes, split regexes with ":".')
    parser.add_argument('--with-config', dest='with_config', metavar='path', type=str, default=None, help='If set, loads a solver with given parameter set. Uses default parameters otherwise.')




def run(parser, args):
    if args.with_config:
        print(f'Loading optimizer with config ({args.with_config})...')
    else:
        print('Loading optimizer...')

    print(f'Executing solver for a total of {args.evaluations} evaluations.')

    solver_class = get_solver(args.path)

    instance_argspec = inspect.getargspec(solver_class.__init__)
    instance_args = instance_argspec.args[1:]
    instance_args_default = instance_argspec.defaults

    try:
        from scipy.optimize import minimize
        func = lambda arr: execute(solver_class(*arr), args.evaluations, args.size, args.dimension, verbose=args.verbose)
        optimum_result = minimize(func, instance_args_default)
        if optimum_result.success:
            prints('Obtained optimum values:')
            print('\n'.join(f'{arg}={x}' for arg, x in zip(instance_args, optimum_result.x)))
        else:
            printe(f'Scipy minimize error detected: {optimum_result.message}')
        return optimum_result.success
    except ImportError as e:
        printe('Cannot import package "scipy.optimize". Please run "pip3 install scipy --user" to install this package.')
        return False