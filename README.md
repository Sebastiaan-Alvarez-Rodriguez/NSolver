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


A project from back in the day on University.
The following implementations exist:
 1. Simulated Annealing ([learn more](https://en.wikipedia.org/wiki/Simulated_annealing)) 
 2. Genetic Algorithm ([learn more](https://en.wikipedia.org/wiki/Genetic_algorithm))

A lot of cleanup is still required here:
The encompassing comperator framework was of such poor quality that I will rebuild it from scratch.

