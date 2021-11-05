from nsolver.optimizer import is_optimizer
from nsolver.utils.printer import *

__name__ = 'run'
__help__ = 'Execute a solver.'


def build_cli(parser):
    parser.add_argument('path', metavar='path', type=str, help='Path to solver implementation.')
    parser.add_argument('--evaluations', metavar='amount', type=int, default=10000, help='Amount of evaluations per parser (default=10,000). A higher number takes longer.')
    parser.add_argument('--filters', metavar='filter[:filter]...]', type=str, default=None, help='If set, only executes solvers with names matching given regex. To supply multiple regexes, split regexes with ":".')



def run(parser, args):
    print('Loading optimizer...')
    success, opimizer = is_optimizer(args.path)
    if not success:
        printe(f'Could not load optimizer at "{args.path}')
        return False

    print(f'Executing parser for {args.evaluations} evaluations.')
    return True