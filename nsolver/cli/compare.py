from nsolver.utils.printer import *
from nsolver import get_solvers

from .run import execute
import concurrent.futures
import itertools
import operator

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
    parser.add_argument('--threads', metavar='number', type=int, default=4, help='Sets amount of executions to perform in parallel (default=4).')


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
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures_execute = {k: executor.submit(execute, v[1](), args.evaluations, args.size, args.dimension, verbose=args.verbose) for k,v in solvers.items() if v[0]}
        solution_results = {k: v.result() for k,v in futures_execute.items()}


    print('Ranking:')
    ranked_results = sorted(solution_results.items(), key=lambda x: x[1][1:2:-1])
    print(ranked_results)
    for idx, (k,v) in enumerate(ranked_results):
        print(f'{idx:03d} - {k}:')
        printc('fitness = ', Color.CAN, f'{v[2]:.03f}', Color.CLR, '. time = ', Color.PRP, f'{v[1]:03f}s', Color.CLR, '. Found solution = ', *((Color.GRN, 'yes') if v[2]==0.0 else (Color.RED, 'no')))
        print_solution(v[0], args.size, args.dimension)
    return True