import numpy as np



def evaluate(cube, dim=3):
    if dim == 2:
        return evaluate_square(cube)
    elif dim == 3:
        return evaluate_cube(cube)
    else:
        raise NotImplementedError(f'No eval function found for dim={dim}')


def evaluate_square(square):
    '''Fitness function for a magic square: this code takes into account the diagonal sums on each square slice.
    Args:
        square (list(int)): the solution vector that represents a magic cube.
    Returns:
        double: the error value of the input solution vector.
        The mean squared error (MSE) of all each row, column, diagonal and space diagonal sum to the magic constant is computed.
    Author: Koen van der Blom, Hao Wang, Sander van Rijn.'''

    square, magic_constant = _verify_square(square)
    errors = _calc_square_errors(square, magic_constant)
    mse = np.mean(errors)
    return mse


def evaluate_cube(cube):
    '''Fitness function for a magic square: this code takes into account the diagonal sums on each square slice.
    Args:
        cube (list(int)): the solution vector that represents a magic cube.
    Returns:
        double: the error value of the input solution vector.
        The mean squared error (MSE) of all each row, column, diagonal and space diagonal sum to the magic constant is computed.
    Author: Koen van der Blom, Hao Wang, Sander van Rijn.'''

    cube, magic_constant = _verify_cube(cube)
    errors = np.concatenate([_calc_cube_errors(cube, magic_constant, diag=True), _calc_space_square_diag_errors(cube, magic_constant)])
    mse = np.mean(errors)
    return mse



def is_solution(cube, dim=3):
    return evaluate(cube, dim=dim) == 0.0 


def raise_representation_error(num_numbers, required_numbers, representation):
    missing = required_numbers - set(representation)
    not_belong = set(representation) - required_numbers
    not_belong = None if not any(not_belong) else not_belong
    raise ValueError(f'''Invalid representation! The solution should be a permutation of 1,...,{num_numbers}
Missing numbers: {missing}
Numbers that do not belong: {not_belong}
Array: {representation}''')


def _verify_square(square):
    n = len(square) ** (1 / 2)
    if np.round(n) ** 2 != len(square):
        raise ValueError('Invalid length! The solution length should be a square number')
    n = int(np.round(n))

    required_numbers = set(range(1, n**2+1))
    if len(set(square) ^ required_numbers) != 0:
        raise_representation_error(n**2, required_numbers, square)

    magic_constant = n * (n ** 2 + 1) / 2
    square = np.array(square).reshape((n, n))
    return square, magic_constant


def _verify_cube(cube):
    n = len(cube) ** (1 / 3)
    if np.round(n) ** 3 != len(cube):
        raise ValueError('Invalid length! The solution length should be a cubic number')
    n = int(np.round(n))

    required_numbers = set(range(1, n**3+1))
    if len(set(cube) ^ required_numbers) != 0:
        raise_representation_error(n**3, required_numbers, cube)

    magic_constant = n * (n**3 + 1) / 2
    cube = np.array(cube).reshape((n, n, n))
    return cube, magic_constant



def _calc_square_errors(square, magic_constant, diag=True):
    sums = [
        np.sum(square, axis=0),               # columns
        np.sum(square, axis=1),               # rows
    ]
    if diag:
        sums.extend([
            [np.sum(np.diag(square))],            # diagonal 1
            [np.sum(np.diag(np.rot90(square)))],  # diagonal 2
        ])

    return (np.concatenate(sums)-magic_constant)**2


def _calc_cube_errors(cube, magic_constant, diag=True):
    errors = []
    for square in cube:
        errors = np.concatenate((errors,_calc_square_errors(square, magic_constant, diag=diag)))

    for i in range(cube.shape[1]):
        square = cube[:, i, :]
        sums = [
            np.sum(square, axis=1),               # pillars
        ]
        if diag:
            sums.extend([
                [np.sum(np.diag(square))],            # diagonal 1
                [np.sum(np.diag(np.rot90(square)))],  # diagonal 2
            ])
        sums = np.concatenate(sums)
        errors = np.concatenate((errors, (sums-magic_constant)**2))

    if diag:
        for i in range(cube.shape[2]):
            square = cube[:, :, i]
            sums = np.array([
                np.sum(np.diag(square)),            # diagonal 1
                np.sum(np.diag(np.rot90(square))),  # diagonal 2
            ])
            errors = np.concatenate((errors, (sums-magic_constant)**2))

    return errors


def _calc_space_square_diag_errors(cube, magic_constant):
    n = cube.shape[0]
    space_square1 = np.zeros((n, n))
    space_square2 = np.zeros((n, n))
    for i in range(n):
        space_square1[i, :] = np.diag(cube[:, :, i])
        space_square2[i, :] = np.diag(np.rot90(cube[:, :, i]))

    space_diag_sum = np.array([
        np.sum(np.diag(space_square1)),
        np.sum(np.diag(np.rot90(space_square1))),
        np.sum(np.diag(space_square2)),
        np.sum(np.diag(np.rot90(space_square2)))
    ]).flatten()
    errors = (space_diag_sum - magic_constant) ** 2
    return errors
