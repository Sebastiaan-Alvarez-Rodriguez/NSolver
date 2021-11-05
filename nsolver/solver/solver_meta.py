class SolverMeta(object):
    '''Trivial container object, storing a container reference and the path to it.'''
    def __init__(self, path, solver):
        self._path = path
        self._solver = solver

    @property
    def path(self):
        return self._path
    
    @property
    def solver(self):
        return self._solver
    