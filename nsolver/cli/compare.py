from nsolver.utils.printer import *
from nsolver import get_solvers

import itertools

'''
Optimizer comperator for N-cube solvers.

Each optimizer will be tested on the following three problems with a budget
of 10,000 evaluations:
 - 12x12 Magic Square
 - 7x7x7 Semi-Perfect Magic Cube
 - 7x7x7 Perfect Magic Cube

This script will automatically plot (in both .png and .pdf)
 - a comparison of the median convergence for all optimizers per problem
 - a visualization of the best found solution per optimizer and problem
'''


__name__ = 'compare'
__help__ = 'Execute comperator for multiple solvers.'


def build_cli(parser):
    parser.add_argument('paths', metavar='paths', nargs='+', help='Paths to search for solver implementations. Pointed locations can be files or directories. Separate locations using spaces.')
    parser.add_argument('--evaluations', metavar='amount', type=int, default=10000, help='Amount of evaluations per parser (default=10,000). A higher number takes longer.')
    parser.add_argument('--filters', metavar='filter', type=str, default=None, help='If set, only executes solvers with names matching given regex. To supply multiple regexes, split regexes with ":".')


def print_solution(solution, size, dimension):
    for dim in range(size**(dimension-2)):
        for x in range(size):
            print(' '.join(f'{solution[dim*size*size+x*size+y]:02d}' for y in range(size)))
        print('')


def run(parser, args):
    print('Searching for solvers...')
    solvers = {}
    for x in itertools.chain(get_solvers(path, filters=args.filters) for path in args.paths):
        solvers.update(x)

    prints(f'Found {sum(1 for x in solvers.values() if x[0])} solver(s).')
    if args.verbose:
        for k,v in solvers.items():
            if v[0]:
                print(f'    {k}')

    return True