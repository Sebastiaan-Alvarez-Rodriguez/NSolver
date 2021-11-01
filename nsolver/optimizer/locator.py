import nsolver.utils.fs as fs
import nsolver.utils.importer as importer
import nsolver.optimizer as optimizer
import concurrent.futures
from pathlib import Path


def _find_potential_optimizers(path):
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



def _is_optimizer(module):
    '''Checks whether given Python module contains a valid optimizer.
    A module is an optimizer if and only if there is a function get_optimizer(), which returns a valid `nsolver.Optimizer`.
    Args:
        module (python module): Module to test.
    Returns:
        bool: `True` if given path is an optimizer, `False` otherwise'''
    return False



def _import_and_validate(path):
    '''Tries to import a python module at given path, and validates whether this module contains an optimizer.
    Args:
        path (str or Path): Path to test.
    Returns:
        (bool, Optimizer): `True`, optimizer on success, `False`, `None` on failure.'''
    imported = importer.import_full_path(path)
    if _is_optimizer(imported):
        return True, imported.get_optimizer()
    return False, None



def get_optimizers(path, max_workers=1):
    '''Finds and loads all optimizers found for a given path.
    Args:
        path (str or Path): Path to test. Can be a directory, in which case searching continues recursively.
        max_workers (optional int): Amount of cores to search and load with. Must be >1.
    Returns:
        dict(Path, (bool, Optimizer): A mapping of potential paths to a bool (whether the path contains an optimizer) and an optimizer class (the imported optimizer or `None`). '''
    if max_workers < 1:
        raise ValueError('get_optimizers requires at least 1 worker to search for optimizers.')
    
    vals = {}
    if max_workers == 1:
        for path in _find_potential_optimizers(path):
            vals[path] = _import_and_validate(path)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_get_optimizers = {x: executor.submit(optimizer.get_optimizers, x) for path in _find_potential_optimizers(path)}
            vals = {k: v.result() for k,v in futures_get_optimizers.items()}
    return vals