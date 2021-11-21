# NSolver
Magic N-cube solver framework, focussing on genetic algorithms to solve the magic N-cube problem.

A magic N-cube is a N-dimensional object with axiomial vectors of a constant size X in all dimensions,
where the sum of all X-sized vectors is equal.
E.g. an instance of a magic 3x3 2D cube (a square) would be:
```
2, 7, 6,
9, 5, 1,
4, 3, 8,
```
The sum of every row, column and diagonal vector is `15` in this example.
Note that for diagonals, we only review diagonal vectors of size `3`.


The goal of this project is to compare algorithms to generate a perfect magic N-dimensional cube.
Users can run and compare their own algorithms.

The following example implementations exist:
 + Informed algorithms
     1. Backtracking ([learn more](https://en.wikipedia.org/wiki/Backtracking))
 + Uninformed algorithms
     1. Simulated annealing ([learn more](https://en.wikipedia.org/wiki/Simulated_annealing)) 
     2. Genetic algorithm ([learn more](https://en.wikipedia.org/wiki/Genetic_algorithm))
     3. A random solver



## Requirements
The following dependencies are required for Nsolver:
 1. python>=3.7
 2. numpy>=1.21.3



## Usage
There are 3 primary functions in this framework:
 1. `run` a single solver.
 2. `compare` multiple solvers.
 3. `search` for optimal solver parameters (WIP).

For more information about the options, use:
```bash
python3 nsolver/cli/entrypoint.py -h          # common args
python3 nsolver/cli/entrypoint.py run -h      # run-specific args
python3 nsolver/cli/entrypoint.py compare -h  # compare-specific args
python3 nsolver/cli/entrypoint.py search -h   # search-specific args
```



## Creating your own solver
Custom solvers implement the interface given in [/nsolver/solver/solver.py](/nsolver/solver/solver.py):
```python
def get_solver():
    '''Returns the implemented solver class.'''
    return solver

class Solver(object):
    def __init__(self):
        pass


    @staticmethod
    def from_config(path):
        '''Builds this Solver using given configuration file.
        Args:
            path (str or Path): path to configuration file.
        Returns:
            Solver: Solver implementation created from the configuration file.'''
        raise NotImplementedError('This solver does not support loading from a configuration file.')


    def execute(self, n, dim, evaluations):
        '''Execute this solver for the perfect cube problem, using given args.
        Args:
            n (int): Dimension size. e.g., for n=4 and dim=3, we expect a 4x4x4 cube.
            dim (int): Dimension of magic cube. E.g. for dim=2, must produce a magic square.
            evaluations (int): Maximum number of evaluations to perform.
        Returns:
            list(int): found solution.'''
        raise NotImplementedError
```
For a simple example, see the [random solver](examples/uninformed/random/random.py).
For an example with optimizable parameters, see [simulated annealing solver](examples/uninformed/simulated_annealing/simulated_annealing.py).

When all functions are implemented, the implementation can be debugged using:
```bash
python3 nsolver/cli/entrypoint.py --size 3 --dimension 2 --verbose run <path/to/new/implementation.py>
```

Regular execution can be performed using:
```bash
python3 nsolver/cli/entrypoint.py --size 3 --dimension 2 run <path/to/new/implementation.py> 
```