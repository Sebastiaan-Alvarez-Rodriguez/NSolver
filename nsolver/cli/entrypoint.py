#!/usr/bin/python3

import argparse


def _get_modules():
    import data_deploy.cli.clean as clean
    import data_deploy.cli.deploy as deploy
    import data_deploy.cli.plugin as plugin
    return [deploy, clean, plugin]



def generic_args(parser):
    '''Configure arguments important for all modules (install, uninstall, start, stop) here.'''
    parser.add_argument('--verbose', type=bool, help='Print more verbose output for debugging', action='store_true')



def subparser(parser):
    '''Register subparser modules.'''
    generic_args(parser)
    subparsers = parser.add_subparsers(help='Subcommands', dest='command')
    return [x.subparser(subparsers) for x in _get_modules()]



def cli(parser):
    '''Adds commandline interface (cli) parsers and arguments as needed for each cli module.
    Args:
        parser (argparse.Parser): Base parser to extend.'''
    subparsers = parser.add_subparsers(help='Subcommands', dest='command')

    for module in _get_modules():
        sub = subparsers.add_parser(module.__name__, module.__help__)
        module.cli(sub)


def main():
    parser = argparse.ArgumentParser(
        prog='nsolver',
        formatter_class=argparse.RawTextHelpFormatter,
        description='Compare optimizers on N-cubes.'
    )
    cli(parser)
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