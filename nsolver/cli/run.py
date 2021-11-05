from nsolver.solver import get_solver
from nsolver.utils.printer import *

__name__ = 'run'
__help__ = 'Execute a solver.'


def build_cli(parser):
    parser.add_argument('path', metavar='path', type=str, help='Path to solver implementation.')
    parser.add_argument('--evaluations', metavar='amount', type=int, default=10000, help='Amount of evaluations per parser (default=10,000). A higher number takes longer.')
    parser.add_argument('--filters', metavar='filter[:filter]...]', type=str, default=None, help='If set, only executes solvers with names matching given regex. To supply multiple regexes, split regexes with ":".')



def run(parser, args):
    print('Loading optimizer...')
    try:
        solver_class = get_solver(args.path)
    except Exception as e:
        printe(str(e))
        return False

    print(f'Executing solver for {args.evaluations} evaluations.')

    instance = solver_class()
    instance.execute(args.size, args.dimension, args.evaluations)
    return True