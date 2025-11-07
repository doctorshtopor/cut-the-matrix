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

is_leader = (rank % k2 == 0) #условие: ранг процесса делится без остатка на k2 (на 3)
leader_color = 0 if is_leader else MPI.UNDEFINED

new_comm = comm.Split(color = leader_color, key=rank) #создаю новый коммуникатор только с процессами
#удовлетворяющие условию is_leader

row_group_id = rank // k2 #индекс к какому набору строк (0 или 1 в нашем случае) будет принадлежать строка
#(индекс в массиве rows_per_block)

if is_leader:
    #выделяю место в памяти под каждую полосу 
    recv_row_block = np.empty(rows_per_block[row_group_id] * N, dtype=np.int32)

    if new_comm.Get_rank() == 0:
        #число элементов в каждой полосе
        counts_rows = []
        for i in range(k1):
            count = rows_per_block[i] * N
            counts_rows.append(count)

        #индекс начала следующей полосы
        displs_rows = []
        current_displ = 0
        for i in range(k1):
            displs_rows.append(current_displ)
            current_displ += counts_rows[i]

        
    else:
        counts_rows = None
        displs_rows = None

    new_comm.Scatterv([A if rank == 0 else None, counts_rows, displs_rows, MPI.INT],
                         recv_row_block, root=0)#рассылаю строки процессам 0, k2, 2k2, 3k2...
else:
    recv_row_block = None

str_comm = comm.Split(color=row_group_id, key=rank) #создаю "коммуникататор отдельной полосы"
#процессы (0, 1, 2) главного коммуникатора comm будут получать блоки из первой полосы; (3, 4, 5) - из второй

group_rank = str_comm.Get_rank()

M_local = rows_per_block[row_group_id]
N_local = cols_per_block[group_rank]
recv_local_block = np.empty(M_local * N_local, dtype=np.int32)

#число столбцов для каждого процесса
counts_cols = []
for j in range(k2):
    count = cols_per_block[j]
    counts_cols.append(count)


displs_cols = []
current_displ = 0
for j in range(k2):
    displs_cols.append(current_displ)
    current_displ += cols_per_block[j]
