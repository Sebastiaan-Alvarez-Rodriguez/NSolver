import nsolver.utils.fs as fs
import nsolver.utils.importer as importer
import nsolver.Solver as Solver
import concurrent.futures
from pathlib import Path


def _find_potential_solvers(path):
    '''Searches a path and returns all .py files.
    Args:
        path (Path or str): Searchpath to follow.
    Returns:
        list(Path): All paths to a .py file.'''
    found = []
    for x in fs.ls(path, full_paths=True):
        p = Path(x)
        if p.isfile and str(p).endswith('.py'):
            found.append(p)
        elif p.isdir:
            found.extend(find_projects(path))
    return found



def _is_solver(module):
    '''Checks whether given Python module contains a valid solver.
    A module is an solver if and only if there is a function get_solver(), which returns a valid `nsolver.solver`.
    Args:
        module (python module): Module to test.
    Returns:
        bool: `True` if given path is an solver, `False` otherwise'''
    try:
        solver_class = module.get_solver()
        return issubclass(solver_class, Solver)
    except AttributeError as e:
        return False



def _import_and_validate(path):
    '''Tries to import a python module at given path, and validates whether this module contains an solver.
    Args:
        path (str or Path): Path to test.
    Returns:
        (bool, solver): `True`, solver on success, `False`, `None` on failure.'''
    if not str(path).endswith('.py'):
        raise ValueError(f'Given path does not point to a .py file: {path}')
    imported = importer.import_full_path(path)
    if _is_solver(imported):
        return True, imported.get_solver()
    return False, None


def is_solver(path):
    '''Tries to import a python module at given path, and validates whether this module contains an solver.
    Args:
        path (str or Path): Path to test.
    Returns:
        (bool, solver): `True`, solver on success, `False`, `None` on failure.'''
    return _import_and_validate(path)


def get_solvers(path, max_workers=1):
    '''Finds and loads all solvers found for a given path.
    Args:
        path (str or Path): Path to test. Can be a directory, in which case searching continues recursively.
        max_workers (optional int): Amount of cores to search and load with. Must be >1.
    Returns:
        dict(Path, (bool, solver): A mapping of potential paths to a bool (whether the path contains an solver) and an solver class (the imported solver or `None`). '''
    if max_workers < 1:
        raise ValueError('get_solvers requires at least 1 worker to search for solvers.')
    
    vals = {}
    if max_workers == 1:
        for path in _find_potential_solvers(path):
            vals[path] = _import_and_validate(path)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_get_solvers = {x: executor.submit(solver.get_solvers, x) for path in _find_potential_solvers(path)}
            vals = {k: v.result() for k,v in futures_get_solvers.items()}
    return vals