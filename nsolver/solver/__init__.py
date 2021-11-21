from .solver import Solver
from .solver_meta import SolverMeta
from .locator import get_solvers, is_solver, get_solver

from .fitness import evaluate, evaluate_correct, is_solution
from .fitness import (sum_closest_available, sum_closest, sum_available,
sum_vector, sum_row, sum_col, sum_diagonal, calc_magic_constant)