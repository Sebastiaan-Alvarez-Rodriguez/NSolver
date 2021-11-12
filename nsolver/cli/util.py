from nsolver.utils.printer import *


def print_solution(solution, size, dimension):
    for dim in range(size**(dimension-2)):
        for x in range(size):
            print(' '.join(f'{solution[dim*size*size+x*size+y]:02d}' for y in range(size)))
        print('')