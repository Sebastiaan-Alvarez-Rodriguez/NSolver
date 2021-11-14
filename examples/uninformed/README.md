# Uninformed Solvers
Uninformed solvers are special variants of solvers that use *uninformed* algorithms to solve the magic cube problem:

They have no knowledge of the actual problem they are supposed to solve.
Instead, they commonly have access to a (black-box) function describing how close an input solution is to the desired solution.
This way, programmers define a description of the desired solution, rather than a way of actually solving the problem.
The algorithm is in charge of exploring the search space of possible solution.

The main advantage of uninformed solvers is that they can be reused with completely different problems.
Of course, the programmer has to produce a black-box function for the new problem.

A disadvantage is that solution exploration is often computationally expensive and inefficient, relying on randomized values.