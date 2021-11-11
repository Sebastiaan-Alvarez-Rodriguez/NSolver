from nsolver import get_solver, is_solution, evaluate
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


def execute(instance, evaluations, size, dimension, verbose=False):
    t0 = time.time()
    solution = instance.execute(size, dimension, evaluations, verbose)
    t_delta = time.time() - t0

    return solution, t_delta, evaluate(solution, dim=dimension)


def run(parser, args):
    if args.with_config:
        print(f'Loading optimizer with config ({args.with_config})...')
    else:
        print('Loading optimizer...')

    print(f'Executing solver for {args.evaluations} evaluations.')

    solver_class = get_solver(args.path)
    instance = solver_class.from_config(args.with_config) if args.with_config else solver_class() 

    solution, t_delta, fitness = execute(instance, args.evaluations, args.size, args.dimension, verbose=args.verbose)

    print_solution(solution, args.size, args.dimension)
    if is_solution(solution, dim=args.dimension):
        prints(f'({t_delta:03f}s) Obtained value is a magic cube.')
    else:
        printw(f'({t_delta:03f}s) Obtained value is not a magic cube.')
    return True