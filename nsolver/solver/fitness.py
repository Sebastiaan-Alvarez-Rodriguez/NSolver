import numpy as np



def evaluate(cube, n, dim=3):
    if dim == 2:
        return evaluate_square(cube, n)
    elif dim == 3:
        return evaluate_cube(cube, n)
    else:
        raise NotImplementedError(f'No eval function found for dim={dim}')


def evaluate_correct(cube, n=None, dim=3):
    _basic_checks(cube, dim=dim)
    if n == None:
        n = int(len(cube)**(1/dim))

    magic_constant = calc_magic_constant(n, dim)
    # Check all non-diagonal vectors
    for d in range(dim):
        if any(sum_vector(cube, n, dim, x, d) != magic_constant for x in range(1, n)):
            return False

    # TODO: Check all diagonal vectors
    if evaluate(cube, n, dim=dim) != 0.0:
        return False
    return True

def sum_vector(grid, n, dim, row_idx, axis):
    '''Given a flat array representing a n^dim cube, returns the sum of a vector of the cube for given row and dimension.
    Args:
        grid (list(int) or np.array): A single list representing a `dim`-dimensional grid. Every `n` items form a row.
        n (int): Length of each vertex in the cube.
        dim (int): Amount of dimensions for our grid. E.g. for `dim=2`, grid is a 2D-square.
        row_idx (int): The row-number to sum.
        axis (int): Provides the dimension on which we sum a row. 
    Returns: sum of requested row.'''
    if axis == 0: 
        # axis = 0 --> x-axis (rows). Rows are sequential numbers in the array, starting on any idx % n == 0.
        return np.sum(grid[row_idx*n:row_idx*n+n:1])
    if axis == 1:
        # axis = 1 --> y-axis (cols). Cols are numbers in the array with n-length jumps between them. 
        offset = (row_idx // n) * n**2 + row_idx % n
        return np.sum(grid[offset:(offset+n**2):n])
    if axis == 2: 
        # axis = 2 --> z-axis. Z-Rows are numbers in the array with n**2-length jumps between them. 
        offset = (row_idx // n**2) * n**3 + row_idx % n**2
        return np.sum(grid[offset:offset+n**3:n**2])

    offset = (row_idx // n**axis) * n**(axis+1) + row_idx % n**axis
    return np.sum(grid[offset:offset+n**(axis+1):n**axis])


def sum_available(grid, n, dim, row_idx, axis, cur_idx):
    '''Determines whether it is possible to compute the row_idx'th vector sum in a partially constructed array.
    Args:
        grid (list(int) or np.array): A single list representing a `dim`-dimensional grid.
        n (int): Length of each vertex in the cube.
        dim (int): Amount of dimensions for our grid. E.g. for `dim=2`, grid is a 2D-square.
        row_idx (int): The row-number to check for availability.
        axis (int): Provides the dimension on which we sum a row.
        cur_idx (int): Current index in the grid (assumed not filled yet).
    Returns: 
        bool: `True` if we can apply this sum on the current grid, `False` otherwise.'''
    if cur_idx == 0:
        return False
    if axis == 0:
        return cur_idx >= row_idx*n+n
    if axis == 1:
        offset = (row_idx // n) * n**2 + row_idx % n
        return cur_idx >= offset+n**2
    if axis == 2:
        offset = (row_idx // n**2) * n**3 + row_idx % n**2
        return cur_idx >= offset+n**3
    else:
        offset = (row_idx // n**axis) * n**(axis+1) + row_idx % n**axis
        return cur_idx >= offset+n**(axis+1)


def sum_closest_available(grid, n, dim, axis, cur_idx):
    '''See sum_available(grid, n, dim, row_idx, axis, cur_idx).
    Automatically fills in the closest row_idx for partially constructed grid.
    Returns:
        int, bool: The closest row index; whether the sum is available in given partially constructed cube.'''
    closest_row_idx = sum_closest(grid, n, dim, axis, cur_idx)
    return closest_row_idx, sum_available(grid, n, dim, closest_row_idx, axis, cur_idx)


def sum_closest(grid, n, dim, axis, cur_idx):
    '''Given a flat array representing a n^dim cube and an index, computes the closest row index.
    Args:
        grid (list(int) or np.array): A single list representing a `dim`-dimensional grid.
        n (int): Length of each vertex in the cube.
        dim (int): Amount of dimensions for our grid. E.g. for `dim=2`, grid is a 2D-square.
        axis (int): The dimension to find the closest row index for.
        cur_idx (int): Current index in the grid (assumed not filled yet).
    Returns:
        int: The closest row index for given axis. This row index may or may not be available yet,
             depending on whether the pointed row is completely filled or not (check with sum_available()).'''
    ans = _sum_closest(grid, n, dim, axis, cur_idx)
    return ans if ans >= 0 else 0

def _sum_closest(grid, n, dim, axis, cur_idx):
    last_filled_idx = cur_idx-1
    if axis == 0:
        return last_filled_idx // n
    if axis == 1:
        offset = (last_filled_idx // n**2) * n
        local_square_idx = last_filled_idx % n**2
        return offset + (0 if local_square_idx <= n*(n-1) else local_square_idx - n*(n-1))
    if axis == 2:
        offset = (last_filled_idx // n**3) * n**2
        local_square_idx = last_filled_idx % n**3
        return offset + (0 if local_square_idx <= (n**2)*(n-1) else local_square_idx - (n**2)*(n-1))
    else:
        offset = (last_filled_idx // n**(axis+1)) * n**axis
        local_square_idx = last_filled_idx % n**(axis+1)
        return offset + (0 if local_square_idx <= (n**(axis-1))*(n-1) else local_square_idx - (n**(axis-1))*(n-1))
 

def sum_row(grid, n, dim, row_idx):
    '''Computes the sum of a single row.
    Args:
        grid (list(int) or np.array): A single list representing a `dim`-dimensional grid.
        n (int): Length of each vertex in the cube.
        dim (int): Amount of dimensions for our grid. E.g. for `dim=2`, grid is a 2D-square.
        row_idx (int): The row-number to sum.
    Returns: sum of requested row.'''
    return sum_vector(grid, n, dim, row_idx, 0)


def sum_col(grid, n, dim, col_idx):
    '''Computes the sum of a single column.
    Args:
        grid (list(int) or np.array): A single list representing a `dim`-dimensional grid.
        n (int): Length of each vertex in the cube.
        dim (int): Amount of dimensions for our grid. E.g. for `dim=2`, grid is a 2D-square.
        col_idx (int): The col-number to sum.
    Returns: sum of requested col.'''
    return sum_vector(grid, n, dim, col_idx, 1)


def sum_diagonal(grid, n, dim, diagonal_idx, upper=True):
    # TODO: Add inner diagonals for >2 dimensional cubes.
    raise NotImplementedError(f'Did not implement diagonal walks yet')
    offset = diagonal_idx * n**dim
    if upper:
        return np.sum(grid[offset:offset+n**2:n+1])
    return np.sum(grid[offset+n-1:offset+n**2:n-1])


def calc_magic_constant(n, dim):
    return n*(n**dim+1) / 2


def calc_num_k_cubes(k, n):
    '''Returns the number of k-cubes inside an n-cube.
    E.g., for k=2 and n=3, returns 6. There are 6 squares in a cube.
    From: https://www.math.brown.edu/tbanchof/Beyond3d/chapter4/section07.html''' 
    pass


def evaluate_square(square, n):
    '''Fitness function for a magic square: this code takes into account the diagonal sums on each square slice.
    Args:
        square (list(int)): Solution vector that represents a magic cube.
        n (int): dimension length. e.g., for n=2, we expect a 2x2 square.
    Returns:
        double: the error value of the input solution vector.
        The mean squared error (MSE) of all each row, column, diagonal and space diagonal sum to the magic constant is computed.
    Author: Koen van der Blom, Hao Wang, Sander van Rijn.'''
    square, magic_constant = _verify_square(square, n)
    errors = _calc_square_errors(square, magic_constant)
    mse = np.mean(errors)
    return mse


def evaluate_cube(cube, n):
    '''Fitness function for a magic square: this code takes into account the diagonal sums on each square slice.
    Args:
        cube (list(int)): the solution vector that represents a magic cube.
        n (int): dimension length. e.g., for n=2, we expect a 2x2 square.
    Returns:
        double: the error value of the input solution vector.
        The mean squared error (MSE) of all each row, column, diagonal and space diagonal sum to the magic constant is computed.
    Author: Koen van der Blom, Hao Wang, Sander van Rijn.'''
    cube, magic_constant = _verify_cube(cube, n)
    errors = np.concatenate([_calc_cube_errors(cube, magic_constant, diag=True), _calc_space_square_diag_errors(cube, magic_constant)])
    mse = np.mean(errors)
    return mse



def is_solution(cube, n, dim=3):
    return evaluate(cube, n, dim=dim) == 0.0 


def raise_representation_error(num_numbers, required_numbers, representation):
    missing = required_numbers - set(representation)
    not_belong = set(representation) - required_numbers
    not_belong = None if not any(not_belong) else not_belong
    raise ValueError(f'''Invalid representation! The solution should be a permutation of 1,...,{num_numbers}
Missing numbers: {missing}
Numbers that do not belong: {not_belong}
Array: {representation}''')


def _basic_checks(cube, dim=3):
    '''Function to perform basic sanity checks. Raises errors upon detecting faults.'''
    n = len(cube) ** (1 / dim)
    if np.round(n) ** dim != len(cube):
        raise ValueError('Invalid length! The solution length should be a square number')
    n = int(np.round(n))

    required_numbers = set(range(1, n**dim+1))
    if len(set(cube) ^ required_numbers) != 0:
        raise_representation_error(n**dim, required_numbers, cube)


def _verify_square(square, n):
    _basic_checks(square, dim=2)

    magic_constant = calc_magic_constant(n, 2)
    square = np.array(square).reshape((n, n))
    return square, magic_constant


def _verify_cube(cube, n):
    _basic_checks(square, dim=3)

    magic_constant = calc_magic_constant(n, 3)
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
