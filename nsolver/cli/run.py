

__name__ = 'run'
__help__ = 'Execute a solver.'


def cli(parser):
    parser.add_argument('--evaluations', metavar='amount', type=int, default=10000, help='Amount of evaluations per parser (default=10,000). A higher number takes longer.')
    parser.add_argument('--key-path', dest='key_path', type=str, default=None, help='Path to ssh key to access nodes.')
    parser.add_argument('--filters', metavar='filter', type=str, default=None, help='If set, only executes solvers with names matching given regex. To supply multiple regexes, split regexes with ":".')
    parser.add_argument('--path', metavar='path', nargs=str, help='Path to solver implementation.')



def run(evaluations, filters, paths, verbose=False):
    print('Searching for optimizers...')
    optimizers = get_optimizers(paths)
    if verbose:
        for k,v in optimizers.items():
            print(f'    Found {len(v):03d} optimizers in path {k}')
    prints(f'    Found {sum(x for x in optimizers.values())} optimizers.')
    return True