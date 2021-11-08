from nsolver import get_solver, is_solution
from nsolver.utils.printer import *

import time

__name__ = 'run'
__help__ = 'Execute a solver.'


def build_cli(parser):
    parser.add_argument('path', metavar='path', type=str, help='Path to solver implementation.')
    parser.add_argument('--evaluations', metavar='amount', type=int, default=10000, help='Amount of evaluations per parser (default=10,000). A higher number takes longer.')
    parser.add_argument('--filters', metavar='filter[:filter]...]', type=str, default=None, help='If set, only executes solvers with names matching given regex. To supply multiple regexes, split regexes with ":".')
    parser.add_argument('--with-config', dest='with_config', metavar='path', type=str, default=None, help='If set, loads a solver with given parameter set. Uses default parameters otherwise.')


def print_solution(solution, size, dimension):
    for dim in range(size**(dimension-2)):
        for x in range(size):
            print(' '.join(f'{solution[dim*size*size+x*size+y]:02d}' for y in range(size)))
        print('')


def run(parser, args):
    try:
        if args.with_config:
            print(f'Loading optimizer with config ({args.with_config})...')
        else:
            print('Loading optimizer...')

        solver_class = get_solver(args.path)
        print(f'Executing solver for {args.evaluations} evaluations.')

        instance = solver_class.from_config(args.with_config) if args.with_config else solver_class() 
        t0 = time.time()
        solution = instance.execute(args.size, args.dimension, args.evaluations)
        t_delta = time.time() - t0

        print_solution(solution, args.size, args.dimension)
        if is_solution(solution, dim=args.dimension):
            prints(f'({t_delta:03f}s) Obtained value is a magic cube.')
        else:
            printw(f'({t_delta:03f}s) Obtained value is not a magic cube.')
        return True
    except Exception as e:
        printe(str(e))
        return False
