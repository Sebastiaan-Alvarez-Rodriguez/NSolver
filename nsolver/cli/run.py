from nsolver import get_solver, is_solution
from nsolver.utils.printer import *

__name__ = 'run'
__help__ = 'Execute a solver.'


def build_cli(parser):
    parser.add_argument('path', metavar='path', type=str, help='Path to solver implementation.')
    parser.add_argument('--evaluations', metavar='amount', type=int, default=10000, help='Amount of evaluations per parser (default=10,000). A higher number takes longer.')
    parser.add_argument('--filters', metavar='filter[:filter]...]', type=str, default=None, help='If set, only executes solvers with names matching given regex. To supply multiple regexes, split regexes with ":".')


def print_solution(solution, size, dimension):
    for dim in range(dimension-1):
        for x in range(size):
            s = ' '.join(f'{solution[x*size+y]:02d}' for y in range(size))
            print(s)


def run(parser, args):
    print('Loading optimizer...')
    try:
        solver_class = get_solver(args.path)
    except Exception as e:
        printe(str(e))
        return False

    print(f'Executing solver for {args.evaluations} evaluations.')

    instance = solver_class()
    solution = instance.execute(args.size, args.dimension, args.evaluations)

    print_solution(solution, args.size, args.dimension)
    if is_solution(solution, dim=args.dimension):
        prints('Obtained value is a magic cube.')
    else:
        printw('Obtained value is not a magic cube.')
    return True