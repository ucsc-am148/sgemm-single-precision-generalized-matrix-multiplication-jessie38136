"""Student kernels for the SGEMM autograder assignment.

You implement K2 (GMEM coalescing), K3 (shared-memory blocking), K4 (1D
register tiling), and K5 (2D register tiling) inside this file. The launch
wrappers, tile-size constants, and signatures are provided — you only edit
the kernel bodies marked TODO.

K1 (naive) is given as a worked example so you have a reference for the
numba.cuda @cuda.jit signature every kernel must match.

To check correctness locally before submitting:
    python sanity_check.py

To submit: push your edits to the main branch of this assignment repo.
Each push that touches kernels.py triggers the autograder, which runs
on a Modal A100 40GB and posts your grade as a comment on the commit.
You have 5 graded submissions per assignment.
"""
import math

from numba import cuda, float32


# ── Tile constants ──────────────────────────────────────────────────
# These are tied to the launch shapes the autograder will use. Do not
# change them; the run_kN wrappers below depend on these values.

BLOCKSIZE = 32          # K1 + K2 tile

# K3 tile sizes
BM3, BN3, BK3 = 32, 32, 32

# K4 tile sizes
BM4, BN4, BK4 = 64, 64, 8
TM4 = 8

# K5 tile sizes
BM5, BN5, BK5 = 128, 128, 8
TM5, TN5 = 8, 8


# ── K1: naive (worked example, do not edit) ─────────────────────────

@cuda.jit
def sgemm_naive(A, B, C, M, N, K):
    """K1: one thread per output element. No tiling, no shared memory.
    Provided so you have a working numba.cuda kernel for reference.
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K2: GMEM coalescing (TODO) ──────────────────────────────────────

@cuda.jit
def sgemm_coalesced(A, B, C, M, N, K):
    """K2: rewrite K1 so that 32 threads in a warp end up writing to 32
    *consecutive columns* of C (and reading 32 consecutive elements of B).
    The arithmetic is identical to K1

    Launch shape (run_k2 below uses this):
        block = (BLOCKSIZE * BLOCKSIZE,)        # 1024 threads, 1D
        grid  = (ceil(M / BLOCKSIZE), ceil(N / BLOCKSIZE))

    With a 1D block of 1024 threads, threadIdx.x runs 0..1023.
    Derive (row_in_tile, col_in_tile) from threadIdx.x using integer division
    and modulo by BLOCKSIZE.
    Be careful which one indexes the column.
    """
    tid = cuda.threadIdx.x

    row_in_tile = tid // BLOCKSIZE
    col_in_tile = tid % BLOCKSIZE

    row = cuda.blockIdx.x * BLOCKSIZE + row_in_tile
    col = cuda.blockIdx.y * BLOCKSIZE + col_in_tile

    tmp = float32(0.0)

    if row < M and col < N:
        for i in range(K):
            tmp += A[row, i] * B[i, col]
        C[row, col] = tmp
    return


# ── K3: shared-memory cache-blocking (TODO) ─────────────────────────

@cuda.jit
def sgemm_smem(A, B, C, M, N, K):
    """K3: stream the K dimension in chunks of BK3. Each block computes a
            BM3 x BN3 output tile by repeatedly:
        1. cooperatively loading a BM3 x BK3 slice of A and a BK3 x BN3
           slice of B into shared memory (one element per thread per slice),
        2. cuda.syncthreads(),
        3. dotting the row of As into the column of Bs to update one
           per-thread accumulator,
        4. cuda.syncthreads() before the next K-chunk.

    Launch shape (run_k3 below uses this):
        block = (BM3 * BN3,)                    # 1024 threads, 1D
        grid  = (ceil(M / BM3), ceil(N / BN3))

    Use cuda.shared.array((BM3, BK3), float32) for As and a similar
    (BK3, BN3) for Bs.
    Use 0.0 in the SMEM load when the global index is out of bounds.
    """

    As = cuda.shared.array((BM3, BK3), float32)
    Bs = cuda.shared.array((BK3, BN3), float32)

    tx = cuda.threadIdx.x

    row = cuda.blockIdx.x * BM3 + tx // BN3
    col = cuda.blockIdx.y * BN3 + tx % BN3

    tmp = float32(0.0)

    for t in range((K + BK3 - 1) // BK3):

        a_row = tx // BK3
        a_col = tx % BK3
        global_a_row = cuda.blockIdx.x * BM3 + a_row
        global_k = t * BK3 + a_col

        if global_a_row < M and global_k < K:
            As[a_row, a_col] = A[global_a_row, global_k]
        else:
            As[a_row, a_col] = float32(0.0)

        b_row = tx // BN3
        b_col = tx % BN3
        global_b_row = t * BK3 + b_row
        global_b_col = cuda.blockIdx.y * BN3 + b_col

        if global_b_row < K and global_b_col < N:
            Bs[b_row, b_col] = B[global_b_row, global_b_col]
        else:
            Bs[b_row, b_col] = float32(0.0)

        cuda.syncthreads()

        for i in range(BK3):
            tmp += As[tx // BN3, i] * Bs[i, tx % BN3]

        cuda.syncthreads()

    if row < M and col < N:
        C[row, col] = tmp


# ── K4: 1D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_1d_tile(A, B, C, M, N, K):
    """K4: extend K3 by giving each thread TM4 = 8 rows in a single column
    of the BM4 x BN4 output tile.

    Note: blockIdx.x now indexes COLUMNS of the output.
    The run_k4 wrapper below already accounts for this, but you need to compute the global (row, col)
    start of your block accordingly.

    Launch shape (run_k4 below uses this):
        block = ((BM4 * BN4) // TM4,)           # 512 threads
        grid  = (ceil(N / BN4), ceil(M / BM4))  # x = col, y = row

    Cooperative loads here are tidy: A's tile is BM4 x BK4 = 512 elements,
    B's tile is BK4 x BN4 = 512 elements, and you have 512 threads so
    exactly one element per thread per tile (so no inner-load loop)

    Use cuda.local.array(TM4, float32) for the per-thread accumulator array.
    Initialize all entries to 0.0 before the K-loop.
    """
    As = cuda.shared.array((BM4, BK4), float32)
    Bs = cuda.shared.array((BK4, BN4), float32)

    tx = cuda.threadIdx.x   # 0..511

    # blockIdx.x → 列方向，blockIdx.y → 行方向
    # 每个线程负责 TM4 行、1 列
    thread_row = tx // BN4          # tile 内"哪一列组" → 负责的行起点偏移 [0, BM4/TM4)
    thread_col = tx % BN4           # tile 内列 [0, BN4)

    # 该 block 负责的输出全局起始坐标
    block_row_start = cuda.blockIdx.y * BM4   # 注意 y → 行
    block_col_start = cuda.blockIdx.x * BN4   # 注意 x → 列

    # 全局列（该线程只负责一列）
    global_col = block_col_start + thread_col

    # 累加器：TM4 个输出行
    acc = cuda.local.array(TM4, float32)
    for i in range(TM4):
        acc[i] = float32(0.0)

    for t in range((K + BK4 - 1) // BK4):

        # 加载 A tile (BM4 x BK4)，
        # tx: 0..511，A tile 有 64*8=512 个元素
        a_row = tx // BK4                          # [0, BM4)
        a_col = tx % BK4                           # [0, BK4)
        global_a_row = block_row_start + a_row
        global_a_col = t * BK4 + a_col

        if global_a_row < M and global_a_col < K:
            As[a_row, a_col] = A[global_a_row, global_a_col]
        else:
            As[a_row, a_col] = float32(0.0)

        # 加载 B tile (BK4 x BN4)
        b_row = tx // BN4                          # [0, BK4)
        b_col = tx % BN4                           # [0, BN4)
        global_b_row = t * BK4 + b_row
        global_b_col = block_col_start + b_col

        if global_b_row < K and global_b_col < N:
            Bs[b_row, b_col] = B[global_b_row, global_b_col]
        else:
            Bs[b_row, b_col] = float32(0.0)

        cuda.syncthreads()

        # 计算：该线程负责 TM4 行，dot 进各自 acc
        for k in range(BK4):
            b_val = Bs[k, thread_col]
            for m in range(TM4):
                
                As_row = thread_row * TM4 + m # 该线程负责的第 m 行在 tile 内的行号
                acc[m] += As[As_row, k] * b_val

        cuda.syncthreads()

    # 写回 
    for m in range(TM4):
        global_row = block_row_start + thread_row * TM4 + m
        if global_row < M and global_col < N:
            C[global_row, global_col] = acc[m]
    return


# ── K5: 2D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_2d_tile(A, B, C, M, N, K):
    """K5: extend K4 to a TM5 x TN5 = 8 x 8 register tile per thread.
    Inside the inner-k loop, cache TM5 As values and TN5 Bs values into
    register arrays, then do the TM5 x TN5 outer-product update.

    Launch shape (run_k5 below uses this):
        block = ((BM5 * BN5) // (TM5 * TN5),)   # 256 threads
        grid  = (ceil(N / BN5), ceil(M / BM5))

    Cooperative loads now need a stride loop: the tile has more elements
    (BM5 * BK5 = 1024) than the block has threads (256), so each thread
    loads BM5 * BK5 / 256 = 4 elements of A per K-chunk and similarly for B.
    Pick the per-thread row stride so that consecutive threads touch
    consecutive memory addresses (= coalesced GMEM loads).

    For accumulators, use cuda.local.array((TM5, TN5), float32).
    Numba supports tuple-shaped local arrays!
    """
    As = cuda.shared.array((BM5, BK5), float32)
    Bs = cuda.shared.array((BK5, BN5), float32)

    tx = cuda.threadIdx.x   # 0..255

    # 每线程负责 TM5 行 × TN5 列的输出子块
    # 每 block 在列方向有 BN5/TN5=16 个线程，行方向有 BM5/TM5=16 个线程
    thread_row = tx // (BN5 // TN5)    # [0, 16)
    thread_col = tx % (BN5 // TN5)     # [0, 16)

    block_row_start = cuda.blockIdx.y * BM5
    block_col_start = cuda.blockIdx.x * BN5

    # 累加器：TM5 × TN5
    acc = cuda.local.array((TM5, TN5), float32)
    for m in range(TM5):
        for n in range(TN5):
            acc[m, n] = float32(0.0)

    # 加载 A 时的行步长：256 threads，BM5*BK5=1024 元素，每线程加载 4 个
    # 让连续线程访问连续列（coalesced）：
    #   thread tx 负责 A[tx // BK5, tx % BK5], [tx // BK5 + stride, ...], ...
    A_load_stride = (BM5 * BK5) // (cuda.blockDim.x)   # = 4 rows per thread?
    # 更清晰的写法：把 1024 个元素铺开，每线程按 blockDim.x 步长跳
    num_threads = cuda.blockDim.x   # 256

    for t in range((K + BK5 - 1) // BK5):

        # 加载 A tile (BM5 x BK5 = 1024 元素)，每线程加载 4 个
        for load_idx in range((BM5 * BK5) // num_threads):
            elem_idx = load_idx * num_threads + tx
            a_row = elem_idx // BK5
            a_col = elem_idx % BK5
            global_a_row = block_row_start + a_row
            global_a_col = t * BK5 + a_col
            if global_a_row < M and global_a_col < K:
                As[a_row, a_col] = A[global_a_row, global_a_col]
            else:
                As[a_row, a_col] = float32(0.0)

        # 加载 B tile (BK5 x BN5 = 1024 元素)，每线程加载 4 个
        for load_idx in range((BK5 * BN5) // num_threads):
            elem_idx = load_idx * num_threads + tx
            b_row = elem_idx // BN5
            b_col = elem_idx % BN5
            global_b_row = t * BK5 + b_row
            global_b_col = block_col_start + b_col
            if global_b_row < K and global_b_col < N:
                Bs[b_row, b_col] = B[global_b_row, global_b_col]
            else:
                Bs[b_row, b_col] = float32(0.0)

        cuda.syncthreads()

        # 计算 TM5 × TN5 外积累加
        for k in range(BK5):
            # 缓存进寄存器
            a_reg = cuda.local.array(TM5, float32)
            b_reg = cuda.local.array(TN5, float32)

            for m in range(TM5):
                a_reg[m] = As[thread_row * TM5 + m, k]
            for n in range(TN5):
                b_reg[n] = Bs[k, thread_col * TN5 + n]

            # 外积更新
            for m in range(TM5):
                for n in range(TN5):
                    acc[m, n] += a_reg[m] * b_reg[n]

        cuda.syncthreads()

    # 写回
    for m in range(TM5):
        for n in range(TN5):
            global_row = block_row_start + thread_row * TM5 + m
            global_col = block_col_start + thread_col * TN5 + n
            if global_row < M and global_col < N:
                C[global_row, global_col] = acc[m, n]
    return


# ── Launch wrappers (provided — do not edit) ────────────────────────

def run_k1(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE, BLOCKSIZE)
    sgemm_naive[grid, block](A, B, C, M, N, K)


def run_k2(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE * BLOCKSIZE,)
    sgemm_coalesced[grid, block](A, B, C, M, N, K)


def run_k3(A, B, C, M, N, K):
    grid = (math.ceil(M / BM3), math.ceil(N / BN3))
    block = (BM3 * BN3,)
    sgemm_smem[grid, block](A, B, C, M, N, K)


def run_k4(A, B, C, M, N, K):
    # Axis swap: blockIdx.x indexes columns of C.
    grid = (math.ceil(N / BN4), math.ceil(M / BM4))
    block = ((BM4 * BN4) // TM4,)
    sgemm_1d_tile[grid, block](A, B, C, M, N, K)


def run_k5(A, B, C, M, N, K):
    grid = (math.ceil(N / BN5), math.ceil(M / BM5))
    block = ((BM5 * BN5) // (TM5 * TN5),)
    sgemm_2d_tile[grid, block](A, B, C, M, N, K)


# Graded kernels in the order the rubric uses (1/4 → C, 2/4 → B-, ...).
KERNELS = [
    ("k2_coalesce", run_k2),
    ("k3_smem",     run_k3),
    ("k4_1d_tile",  run_k4),
    ("k5_2d_tile",  run_k5),
]