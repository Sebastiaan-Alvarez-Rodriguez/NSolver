#!/usr/bin/python3

import argparse


def _get_modules():
    import data_deploy.cli.clean as clean
    import data_deploy.cli.deploy as deploy
    import data_deploy.cli.plugin as plugin
    return [deploy, clean, plugin]


def generic_args(parser):
    '''Configure arguments important for all modules (install, uninstall, start, stop) here.'''
    parser.add_argument('--key-path', dest='key_path', type=str, default=None, help='Path to ssh key to access nodes.')


def subparser(parser):
    '''Register subparser modules.'''
    generic_args(parser)
    subparsers = parser.add_subparsers(help='Subcommands', dest='command')
    return [x.subparser(subparsers) for x in _get_modules()]



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