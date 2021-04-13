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
    
    @property
    def band(self):
        return [self.a1, self.a2, self.a3]

    def task_a(self):
        self.A = func.create_band_matrix(self.N, self.band)
        self.b = func.create_b_vector(self.N, self.f)

    def task_b(self):
        time_jacobi = func.solve_jacobi(self.A, self.b)

        time_gauss_seidl = func.solve_gauss_seidl(self.A, self.b)

        func.determine_faster(time_jacobi, time_gauss_seidl)

    def task_c(self):
        self.a1 = 3
        self.a2 = self.a3 = -1

        self.A = func.create_band_matrix(self.N, self.band)

        self.task_b()

    def task_d(self):
        func.solve_lu_factorization(self.A, self.b)

    
        

def main():
    solution = Solution()
    solution.task_a()
    solution.task_b()
    solution.task_c()
    solution.task_d()



if __name__:
    main()