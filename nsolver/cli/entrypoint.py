#!/usr/bin/python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Appends main project root as importpath.


import argparse

def _get_modules():
    import nsolver.cli.compare as compare
    import nsolver.cli.run as run
    return [compare, run]



def generic_args(parser):
    '''Configure arguments important for all modules (install, uninstall, start, stop) here.'''
    parser.add_argument('--verbose', help='Print more verbose output for debugging', action='store_true')
    parser.add_argument('--size', type=int, help='Magic cube size (default=3)', default=3)
    parser.add_argument('--dimension', type=int, help='Magic cube size dimension (default=2)', default=2)


def build_cli(parser):
    '''Adds commandline interface (cli) parsers and arguments as needed for each cli module.
    Args:
        parser (argparse.Parser): Base parser to extend.'''
    subsection = parser.add_subparsers(help='Subcommands', dest='command')
    subparsers = [subsection.add_parser(module.__name__, help=module.__help__) for module in _get_modules()] 
    for subparser, module in zip(subparsers, _get_modules()):
        module.build_cli(subparser)
    return subparsers


def run(parser, subparsers, args):
    '''Searches for the correct CLI module to invoke for given input, and calls its `run()` method.
    Args:
        parser (argparse.Parser): Main parser.
        subparsers (iterable(argparse.Parser): Subparsers, 1 for each module.
        args (argparse.dict): Arguments set through CLI.'''
    for subparser, module in zip(subparsers, _get_modules()):
        if module.__name__ == args.command:
                if args.verbose:
                    return module.run(subparser, args)
                else:
                    try:
                        return module.run(subparser, args)
                    except Exception as e:
                        printe(str(e))
                        return False
    parser.print_help()
    return False


def main():
    parser = argparse.ArgumentParser(
        prog='nsolver',
        formatter_class=argparse.RawTextHelpFormatter,
        description='Compare optimizers on N-cubes.'
    )
    generic_args(parser)
    subparsers = build_cli(parser)
    retval = True

    args = parser.parse_args()

    retval = run(parser, subparsers, args)

    if isinstance(retval, bool):
        exit(0 if retval else 1)
    elif isinstance(retval, int):
        exit(retval)
    else:
        exit(0 if retval else 1)



if __name__ == '__main__':
    main()