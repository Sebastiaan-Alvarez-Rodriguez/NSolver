class OptimizerMeta(object):
    '''Trivial container object, storing a container reference and the path to it.'''
    def __init__(self, path, optimizer):
        self._path = path
        self._optimizer = optimizer

    @property
    def path(self):
        return self._path
    
    @property
    def optimizer(self):
        return self._optimizer
    