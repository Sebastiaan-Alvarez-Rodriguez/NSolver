# NSolver
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

## Parameter Search
Solvers can have `int` parameters and `float` parameters.
Regular optimization functions like `scipy.minimize` are not going to cut it, because they only work for `float` parameters.

[Integer Programming (IP)](https://en.wikipedia.org/wiki/Integer_programming) requires all variables to be `int`.  
Mixed-Integer Programming (MIP) allows variables to be either `int` or `float`.
There are `2` interesting forms of MIP:
 1. Mixed-Integer **Linear** Programming (MILP)
 2. Mixed-Integer **Non-Linear** Programming (MINLP)

### Mixed-Integer Linear Programming (MILP)
The following MILP modules are available in Python:
 1. [python-mip](https://python-mip.readthedocs.io/en/latest/intro.html)
 2. python-pulp


### Mixed-Integer **Non-Linear** Programming (MINLP)
Information: https://scicomp.stackexchange.com/questions/19870/python-solvers-for-mixed-integer-nonlinear-constrained-optimization

Probably need [Mixed-integer linear fractional programming (MILFP)](https://optimization.mccormick.northwestern.edu/index.php/Mixed-integer_linear_fractional_programming_(MILFP))
WIP:
 1. [emcee](https://emcee.readthedocs.io/en/stable/tutorials/quickstart/#quickstart)
 2. [sklearn ML](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)