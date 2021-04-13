import numpy as np
import functions as func

class Solution:
    d = 7
    c = 9
    f = 0
    e = 2
    N = 900 + c * 10 + d

    def __init__(self):
        self.prep_data_task_a()
    
    @property
    def band(self):
        return [self.a1, self.a2, self.a3]

    def prep_data_task_a(self):
        self.a1 = 5 + self.e
        self.a2 = self.a3 = -1

    def prep_data_task_c(self):
        self.a1 = 3
        self.a2 = self.a3 = -1

    def task_a(self):
        self.A = func.create_band_matrix(self.N, self.band)
        self.b = func.create_b_vector(self.N, self.f)

    def task_b(self):
        time_jacobi = func.solve_jacobi(self.A, self.b)

        time_gauss_seidl = func.solve_gauss_seidl(self.A, self.b)

        func.determine_faster(time_jacobi, time_gauss_seidl)

    def task_c(self):
        self.prep_data_task_c()

        self.A = func.create_band_matrix(self.N, self.band)

        self.task_b()

    def task_d(self):
        func.solve_lu_factorization(self.A, self.b)

    def task_e(self):
        N = [100, 500, 1000, 2000, 3000, 4000, 5000]

        self.prep_data_task_a()

        jacobi = []
        gauss_seidl = []

        for n in N:
            self.A = func.create_band_matrix(n, self.band)
            self.b = func.create_b_vector(n, self.f)

            jacobi.append(func.solve_jacobi(self.A, self.b))
            gauss_seidl.append(func.solve_gauss_seidl(self.A, self.b))

        func.plot_times(N, jacobi, gauss_seidl)
        

def main():
    solution = Solution()
    solution.task_a()
    solution.task_b()
    solution.task_c()
    solution.task_d()
    solution.task_e()



if __name__:
    main()