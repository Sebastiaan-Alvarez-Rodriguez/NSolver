from .solver import Solver
from .solver_meta import SolverMeta
from .locator import get_solvers, is_solver, get_solver

from .fitness import evaluate, evaluate_correct, is_solution
from .fitness import sum_row, sum_col, sum_diagonal, sum_vector, calc_magic_constant