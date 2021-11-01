#!/usr/bin/python3

import concurrent.futures
from pathlib import Path
import nsolver.utils.fs as fs
import nsolver.optimizer as optimizer

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

def get_optimizers(paths):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(paths)) as executor:
        futures_get_optimizers = {x: executor.submit(optimizer.get_optimizers, x) for x in paths}
        return {k: v.result() for k,v in futures_get_optimizers.items()}


def run(evaluations, filters, paths, verbose=False):
    print('Searching for optimizers...')
    optimizers = get_optimizers(paths)
    if verbose:
        for k,v in optimizers.items():
            print(f'    Found {len(v):03d} optimizers in path {k}')
    prints(f'    Found {sum(x for x in optimizers.values())} optimizers.')
    return True


def add_args(parser):
    parser.add_argument('--evaluations', metavar='amount', type=int, default=10000, help='Amount of evaluations per parser (default=10,000). A higher number takes longer.')
    parser.add_argument('--filters', metavar='filter', type=str, default=None, help='If set, only executes solvers with names matching given regex. To supply multiple regexes, split regexes with ":".')
    parser.add_argument('--paths', metavar='paths', nargs='+', help='Paths to search for solver implementations. Pointed locations can be files or directories. Separate locations using spaces.')
    parser.add_argument('--verbose', type=bool, help='Print more verbose output for debugging', action='store_true')

    subparsers = parser.add_subparsers(help='Subcommands', dest='command')

    subparsers.add_parser('run', help='Execute solver')
    subparsers.add_parser('compare', help='Execute comperator for multiple solvers')
    parser.add_


def main():
    parser = argparse.ArgumentParser(
        prog='nsolver',
        formatter_class=argparse.RawTextHelpFormatter,
        description='Compare optimizers on N-cubes.'
    )
    add_args(parser)
    retval = True

    args = parser.parse_args()
    retval = run(args.evaluations, args.filterrs, args.paths, verbose=args.verbose)

    if isinstance(retval, bool):
        exit(0 if retval else 1)
    elif isinstance(retval, int):
        exit(retval)
    else:
        exit(0 if retval else 1)



if __name__ == '__main__':
    main()