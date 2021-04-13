import numpy as np
import functions as func

class Solution:
    d = 7
    c = 9
    f = 0
    e = 2
    N = 900 + c * 10 + d

    def __init__(self):
        self.a1 = 5 + self.e
        self.a2 = self.a3 = -1
        self.band = [self.a1, self.a2, self.a3]

    def task_a(self):
        self.A = func.create_band_matrix(self.N, self.band)
        self.b = func.create_b_vector(self.N, self.f)

    def task_b(self):
        time_jacobi = func.solve_jacobi(self.A, self.b)

        time_gauss_seidl = func.solve_gauss_seidl(self.A, self.b)

        if time_jacobi > time_gauss_seidl:
            print(f'Gauss-Seild method was {time_jacobi - time_gauss_seidl} faster')   
        else:
            print(f'Jacobi method was {time_jacobi - time_gauss_seidl} faster')

    def task_c(self):
        pass
        

def main():
    solution = Solution()
    solution.task_a()
    solution.task_b()



if __name__:
    main()