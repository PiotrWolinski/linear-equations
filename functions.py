import numpy as np
from timeit import default_timer as timer

def seperate(func):
    def inner(*args, **kwargs):
        print('=' * 50)
        print(f'Starting {func.__name__}')
        return func(*args, **kwargs)

    return inner

# Creates banded matrix with dimensions equal to size x size and 
# band consisting of 5 elements spread on 5 diagonals
def create_band_matrix(size: int, band: list[int]=None) -> np.ndarray:
    if band is None or len(band) > 3:
        band = [1, 1, 1]
    
    A = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                A[i][j] = band[0]

            if i == j + 1 or i == j - 1:
                A[i][j] = band[1]

            if i == j + 2 or i == j - 2:
                A[i][j] = band[2]
    
    return A

# Creates vector
def create_b_vector(size: int, f: int) -> np.ndarray:
    b = np.zeros((size, 1))

    for i in range(size):
        b[i][0] = np.sin(i * (f + 1))

    return b

@seperate
def solve_jacobi(A: np.ndarray, b: np.ndarray) -> float:
    size = A.shape[0]
    print(f'Matrix size = {size}')
    print('Started solving with Jacobi method...')
    x = np.ones((size, 1))
    res = np.ones((size, 1))
    interrupted = False

    limit = 10 ** -9
    iterations = 0

    start = timer()

    while np.linalg.norm(res) > limit:
        new_x = np.ones((size, 1))

        for i in range(size):

            # first = 0
            # for j in range(i):
            #     first += A[i][j] * x[j][0]

            # second = 0
            # for j in range(i + 1, size):
            #     second += A[i][j] * x[j][0]

            # new_x[i][0] = (b[i][0] - first - second) / A[i][i]
            new_x[i][0] = (b[i][0] - np.sum(A[i][0:i] @ x[0:i][:]) - np.sum(A[i][i+1:size] @ x[i+1:size][:])) / A[i][i]

        res = (A @ new_x) - b
        x = new_x
        iterations += 1

        if iterations > 1000:
            interrupted = True
            break

    total_time = timer() - start

    print('Finished solving with Jacobi method')

    if interrupted:
        total_time = 0
        print('Method was interrupted during execution')

    print(f'Duration: {total_time}')
    print(f'Iterations: {iterations}')

    return total_time

@seperate
def solve_gauss_seidl(A: np.ndarray, b: np.ndarray) -> float:
    size = A.shape[0]
    print(f'Matrix size = {size}')
    print('Started solving with Gauss-Seidl method...')
    x = np.ones((size, 1))
    res = np.ones((size, 1))
    interrupted = False

    limit = 10 ** -9
    iterations = 0

    start = timer()

    while np.linalg.norm(res) > limit:
        new_x = np.ones((size, 1))

        for i in range(size):

            # first = 0
            # for j in range(i):
            #     first += A[i][j] * new_x[j][0]

            # second = 0
            # for j in range(i + 1, size):
            #     second += A[i][j] * x[j][0]

            # new_x[i][0] = (b[i][0] - first - second) / A[i][i]
            new_x[i][0] = (b[i][0] - np.sum(A[i][0:i] @ new_x[0:i][:]) - np.sum(A[i][i+1:size] @ x[i+1:size][:])) / A[i][i]

        res = (A @ new_x) - b
        x = new_x
        iterations += 1

        if iterations > 1000:
            interrupted = True
            break

    total_time = timer() - start
    
    print('Finished solving with Gauss-Seidl method')

    if interrupted:
        total_time = 0
        print('Method was interrupted during execution')

    print(f'Duration: {total_time}')
    print(f'Iterations: {iterations}')

    return total_time

@seperate
def determine_faster(jacobi: float, gauss_seidl: float):
    if jacobi > gauss_seidl:
        print(f'Gauss-Seild method was {jacobi - gauss_seidl} s faster')   
    else:
        print(f'Jacobi method was {jacobi - gauss_seidl} s faster')


@seperate
def solve_lu_factorization(A: np.ndarray, b: np.ndarray):
    m = A.shape[0]

    print(f'Matrix size = {m}')
    print('Started solving with LU factorization...')
    
    U = A.copy()
    L = np.eye(m)

    start = timer()

    for k in range(m):
        for j in range(k + 1, m):
            L[j][k] = U[j][k] / U[k][k]
            U[j][k:m] = U[j][k:m] - (L[j][k] * U[k][k:m])

    total_time = timer() - start

    print('Finished solving with LU factorization')
    print(f'Duration: {total_time}')
    

    