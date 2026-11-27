// =============================================================================
// matmul/04_1d_register_tile.cu
// Matrix multiply, version 3: 1D register tiling (each thread computes TM = 8 outputs)
// -----------------------------------------------------------------------------
// Problem:   C[M,N] = A[M,K] * B[K,N], all fp32 and row-major.
// Requirements: M % 64 == 0, N % 64 == 0 and K % 8 == 0. The kernel has NO bounds checks. A partial
//            block tile in M or N would be left out of C, and a partial tile in K would read past the
//            end of a row of A and B. The repo's launcher in kernels/matmul.cuh is stricter and requires
//            multiples of 128. This file checks the real requirement, in launch_1d_regtile below.
// Idea:      A 64 x 64 block tile of C is computed by 512 threads. Each thread owns one COLUMN of the tile
//            and TM = 8 consecutive ROWS of it, so it computes 8 outputs. The inputs go through shared memory
//            in steps of BK = 8 along K. For each step along K, a thread loads one B value into a register
//            and reuses it for its 8 outputs.
// Work:      2*M*N*K FLOPs. At 2048^3 that is 17.2 GFLOP.
// Changed vs previous version: 03_smem_tiling.cu.
//            - block tile 32 x 32 -> 64 x 64, and 1024 -> 512 threads per block
//            - each thread computes TM = 8 outputs (1 before), held in 8 register accumulators
//            - K step BK = 8 (was 32), so each shared-memory tile is 64 x 8 for A and 8 x 64 for B
//            - the grid is exactly N/64 x M/64, with no ceil_div, because the sizes must be whole tiles
// Why faster: The shared-memory traffic per multiply-add falls. In 03, each FMA needs two shared-memory
//            loads (As[ty][k] and Bs[k][tx]). Here, for each k, a thread loads one Bs value and 8 As values
//            (9 loads) and does 8 FMAs. That is about 1.1 loads per FMA. The Bs value is reused 8 times from
//            a register. Each thread also has 8 independent accumulators, so its 8 FMAs of one step do not
//            wait for each other.
// Run:       ./bin/matmul_04_1d_register_tile                  self-test, then timing
//            ./bin/matmul_04_1d_register_tile --profile        one launch only, for ncu
// Refs:      [BOEHM] (the 1D block-tiling step), [VOLKOV08] (keep reused operands in registers),
//            [PMPP] chapter 5 (tiling), [NV-SMEM] (shared memory)
// =============================================================================

#include "common/helpers.cuh"  // CUDA_CHECK, ceil_div, BenchArgs, time_ms, DeviceBuffer, check_close, ...

// -----------------------------------------------------------------------------
// CPU reference: the answer we trust, computed in double precision
// -----------------------------------------------------------------------------
static void cpu_gemm(int M, int N, int K,
                     const std::vector<float>& A,
                     const std::vector<float>& B,
                     std::vector<float>& C) {
  for (int i = 0; i < M; ++i) {        // every output row
    for (int j = 0; j < N; ++j) {      // every output column
      double acc = 0.0;                // double accumulator: the reference should be more accurate than the GPU
      for (int k = 0; k < K; ++k) {
        acc += (double)A[i * K + k] * B[k * N + j];  // same dot product as the kernel
      }
      C[i * N + j] = (float)acc;       // round once at the end
    }
  }
}


// -----------------------------------------------------------------------------
// The kernel
// -----------------------------------------------------------------------------
// The tile sizes are template parameters, so they are compile-time constants:
//   BM x BN  block tile of C (64 x 64)
//   BK       how many k values are loaded per step (8)
//   TM       how many outputs each thread computes, all in one column (8)
// __launch_bounds__ tells the compiler the largest block size it must support, (BM*BN)/TM = 512 threads,
// so it can plan the number of registers per thread.
template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
    sgemm_1d_blocktile(int M, int N, int K,
                       const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C) {
  // Each block has NUM_THREADS threads. Every thread loads one element of the A tile and one of the B tile
  // per step, so NUM_THREADS must equal the number of elements in each tile.
  constexpr int NUM_THREADS = (BM * BN) / TM;  // 64 * 64 / 8 = 512
  static_assert(BM * BK == NUM_THREADS && BN * BK == NUM_THREADS, "one element per thread per tile load");
  // Shared-memory tiles. __shared__ means one copy per block, visible to all its threads.
  __shared__ float As[BM * BK];  // the A tile: BM rows x BK columns = 64 x 8 floats, 2 KB
  __shared__ float Bs[BK * BN];  // the B tile: BK rows x BN columns = 8 x 64 floats, 2 KB

  // Which output this thread owns. The 512 threads are split into 64 columns (threadCol) and 8 row groups
  // (threadRow). Each thread owns one column, and within it the TM rows threadRow*TM ... threadRow*TM + 7.
  // A warp is 32 consecutive threads of the block (lanes 0..31). With BN = 64, all lanes of a warp share a threadRow.
  const int threadCol = threadIdx.x % BN;  // 0..63: the column of the C tile this thread owns
  const int threadRow = threadIdx.x / BN;  // 0..7: which group of TM rows this thread owns
  // Move the base pointers to this block's tile. blockIdx.y picks the row strip and blockIdx.x the column strip.
  A += blockIdx.y * BM * K;                    // first row of this block's strip of A
  B += blockIdx.x * BN;                        // first column of this block's strip of B
  C += blockIdx.y * BM * N + blockIdx.x * BN;  // top-left element of this block's C tile

  // Each thread loads exactly one element of the A tile and one element of the B tile per step.
  // The A tile is 64 rows x 8 columns, so thread t loads row t / 8 and column t % 8.
  const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;
  // The B tile is 8 rows x 64 columns, so thread t loads row t / 64 and column t % 64.
  const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;

  float acc[TM] = {0.f};  // TM accumulators in registers: one for each output this thread owns
  // Walk along K in steps of BK. bk is the first k of the current tile, counted from 0.
  for (int bk = 0; bk < K; bk += BK) {
    // Load step. One A element and one B element per thread, from global memory into shared memory.
    // A[innerRowA][innerColA] of this tile sits at innerRowA*K + innerColA (A was already moved to the tile).
    // Lanes with consecutive innerColA read consecutive floats of one row, so the loads are coalesced.
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    // B[innerRowB][innerColB] of this tile sits at innerRowB*N + innerColB. Consecutive threads read
    // consecutive columns, so these loads are coalesced too.
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    // First barrier: __syncthreads() makes each thread wait here until every thread of the block has arrived.
    // Every element of both tiles must be in shared memory before anyone reads them. Each thread
    // reads values that other threads loaded, so the barrier is required.
    __syncthreads();
    // Move the pointers to the next tile: BK columns to the right in A, and BK rows down in B.
    A += BK;
    B += BK * N;
    // The 8 steps inside this tile. Each step k multiplies a column of the A tile by a row of the B tile.
    // #pragma unroll writes out all BK iterations of the loop, so k is a compile-time constant. The
    // shared-memory offsets become fixed numbers, and the loop has no counter or branch.
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // One shared-memory load of the B value at row k, column threadCol. It goes into a register, so the
      // 8 FMAs below reuse it without loading again.
      const float b = Bs[k * BN + threadCol];
      // Eight FMAs: acc[i] += A(row threadRow*TM + i, column k) * b. As is stored row-major as 64 x 8, so the
      // element is at (row)*BK + k. All 32 lanes of a warp have the same threadRow, so they read the same As
      // address: a broadcast. Their threadCol values are consecutive, so their Bs reads are consecutive:
      // no bank conflicts.
#pragma unroll
      for (int i = 0; i < TM; ++i) acc[i] += As[(threadRow * TM + i) * BK + k] * b;
    }
    // Second barrier: nobody may start loading the next tile into As and Bs until everyone has finished
    // reading the current one.
    __syncthreads();
  }
  // Write the TM results. Output row threadRow*TM + i of the block tile is at (row)*N + threadCol in C.
  // The loop is unrolled, so the TM stores are straight-line code. Each store is in a different row.
#pragma unroll
  for (int i = 0; i < TM; ++i) C[(threadRow * TM + i) * N + threadCol] = acc[i];
}


// -----------------------------------------------------------------------------
// Host-side launcher: checks the sizes, decides the grid and block shape, then launches the kernel
// -----------------------------------------------------------------------------
static void launch_1d_regtile(int M, int N, int K,
                              const float* A, const float* B, 
                              float* C) {
  constexpr int BM = 64, BN = 64, BK = 8, TM = 8;  // the same tile constants as kernels/matmul.cuh
  // The kernel has no bounds checks, so each size must be a whole number of tiles. Check it here and stop
  // with a message instead of returning wrong numbers.
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    fprintf(stderr, "launch_1d_regtile needs M %% 64 == 0, N %% 64 == 0 and K %% 8 == 0\n");
    std::exit(1);
  }
  // One block per 64 x 64 tile of C. x runs over the N/64 column tiles and y over the M/64 row tiles.
  // Example at 2048^3: 32 x 32 = 1024 blocks of 512 threads each.
  dim3 grid(N / BN, M / BM);
  // 64 * 64 / 8 = 512 threads per block, the NUM_THREADS of the kernel.
  sgemm_1d_blocktile<BM, BN, BK, TM><<<grid, BM * BN / TM>>>(M, N, K, A, B, C);
  CUDA_CHECK_LAST();  // catch an invalid launch configuration right away
}

// -----------------------------------------------------------------------------
// main: self-test on a small problem, then timing on a large one
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  BenchArgs args = BenchArgs::parse(argc, argv);  // read --profile, --iters, --warmup, --csv
  print_device();                                 // show which GPU is being used

  // ---- 1. Self-test: compare against the CPU. Skipped under --profile so ncu sees exactly one launch.
  if (!args.profile) {
    const int M = 256, N = 256, K = 256;  // 256 is a multiple of 64 and of 8, so the sizes are allowed

    std::vector<float> hA((size_t)M * K),
                       hB((size_t)K * N),
                      ref((size_t)M * N);  // host (CPU) arrays
    fill_uniform(hA, 1);  // random inputs in [-1, 1), fixed seed 1
    fill_uniform(hB, 2);  // fixed seed 2

    DeviceBuffer<float> A(hA.size()),
                        B(hB.size()),
                        C((size_t)M * N);  // GPU arrays of the same sizes
    A.upload(hA);  // copy the inputs to the GPU
    B.upload(hB);

    cpu_gemm(M, N, K, hA, hB, ref);  // reference result on the CPU

    launch_1d_regtile(M, N, K, A.ptr, B.ptr, C.ptr);  // run the GPU kernel

    CUDA_CHECK(cudaDeviceSynchronize());              // wait for it to finish, so runtime errors surface here

    std::vector<float> out = C.download();            // copy the result back to the CPU

    // Tolerance: 1e-3 absolute and 1e-3 relative. K = 256 products of values in [-1, 1) differ only by rounding.
    const bool ok = check_close("v3 1d register tile vs CPU", ref.data(), out.data(), out.size(), 1e-3f, 1e-3f);

    if (!ok) return 1;  // a failed self-test makes the program exit with an error code
  }

  // ---- 2. Timing on a large problem.
  const int n = 2048;  // square problem: M = N = K = 2048. 2048 is a multiple of 64 and of 8.

  std::vector<float> hA((size_t)n * n),
                     hB((size_t)n * n);  // CPU copies
  fill_uniform(hA, 3);  // random data (values do not change the speed of a dense matmul)
  fill_uniform(hB, 4);

  DeviceBuffer<float> A((size_t)n * n),
                      B((size_t)n * n),
                      C((size_t)n * n);  // GPU arrays
  A.upload(hA);
  B.upload(hB);

  // time_ms runs the lambda `warmup` times untimed, then `iters` times timed, and returns the average in ms.
  const float ms = time_ms([&] {
                                  launch_1d_regtile(n, n, n, A.ptr, B.ptr, C.ptr); 
                               }, args.warmup, args.iters);

  // A matmul does 2*n^3 floating-point operations. Dividing by seconds and by 1e12 gives TFLOP/s.
  const double tflops = 2.0 * n * n * (double)n / (ms * 1e-3) / 1e12;
  printf("\n%-24s %10.3f ms  %8.4f TFLOP/s\n", "v3 1d register tile", ms, tflops);

  // Optionally append a CSV row (--csv file). An empty path makes this a no-op.
  CsvWriter csv(args.csv);
  csv.row("matmul", "v3_1d_register_tile", "2048x2048x2048", ms, tflops, "TFLOP/s");
  return 0;
}
