from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

M, N = 10, 10
k1, k2 = 2, 3
assert k1 * k2 == size, "numprocs != k1*k2"


if rank == 0:
    A = np.arange(1, M * N + 1, dtype=np.int32).reshape(M, N)
    print("Матрица:\n", A, flush=True)
else:
    A = None


rows_per_block = np.zeros(k1, dtype=np.int32)
cols_per_block = np.zeros(k2, dtype=np.int32)

#распределяю сколько строк будет в каждом блоке
for i in range(k1):
    rows_per_block[i] = M // k1
    if i < M % k1:  #остаток строк делю между первыми блоками
        rows_per_block[i] += 1

#для столбцов аналогично
for j in range(k2):
    cols_per_block[j] = N // k2
    if j < N % k2:  
        cols_per_block[j] += 1
