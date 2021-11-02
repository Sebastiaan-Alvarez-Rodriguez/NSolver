

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


def cli(parser):
    parser.add_argument('--evaluations', metavar='amount', type=int, default=10000, help='Amount of evaluations per parser (default=10,000). A higher number takes longer.')
    parser.add_argument('--key-path', dest='key_path', type=str, default=None, help='Path to ssh key to access nodes.')
    parser.add_argument('--filters', metavar='filter', type=str, default=None, help='If set, only executes solvers with names matching given regex. To supply multiple regexes, split regexes with ":".')
    parser.add_argument('--paths', metavar='paths', nargs='+', help='Paths to search for solver implementations. Pointed locations can be files or directories. Separate locations using spaces.')
