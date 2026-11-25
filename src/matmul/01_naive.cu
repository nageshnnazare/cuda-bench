// =============================================================================
// matmul/01_naive.cu
// Matrix multiply, version 0: the naive kernel (one thread per output element)
// -----------------------------------------------------------------------------
// Problem:   C[M,N] = A[M,K] * B[K,N], all fp32 and row-major.
//            Row-major means element (i, j) of a matrix with C columns sits at index i*C + j.
// Idea:      Each thread computes one element of C as a dot product:
//            C[i][j] = sum over k of A[i][k] * B[k][j]
// Work:      2*M*N*K floating-point operations (one multiply and one add per k).
// Why slow:  threadIdx.x is mapped to ROWS, so the 32 threads of a warp read A at addresses
//            K floats apart. The hardware then issues about 32 separate memory sectors for one
//            load instruction.
// Run:       ./bin/matmul_01_naive                    self-test, then timing
//            ./bin/matmul_01_naive --profile          one launch only, for ncu
//            ./bin/matmul_01_naive --iters 3 --warmup 1   quicker: each call takes ~0.3 s at 2048^3
// Refs:      [PMPP] chapters 3 and 5 (the matrix multiplication example, and tiling), [BOEHM] step 1, [NV-COALESCE]
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
// The gpu kernel
// -----------------------------------------------------------------------------
// Every thread runs this same code; they differ only in their threadIdx and blockIdx values.
__global__ void sgemm_naive(int M, int N, int K,              // matrix sizes: C is M x N, the shared dimension is K
                            const float* A, const float* B,   // inputs in GPU global memory
                            float* C) {                       // output in GPU global memory
  // Global row of the output element this thread computes.
  // blockIdx.x is this block's index in the grid, blockDim.x the block width,
  // threadIdx.x this thread's index inside the block. Together they give a unique index per thread.
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  // Global column, computed the same way from the y dimension.
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  // The grid is rounded up to whole 32x32 blocks, so some threads lie outside C. They must do nothing.
  if (row < M && col < N) {
    float acc = 0.f;  // running dot product for C[row][col]. It lives in a register (private to this thread).

    // Walk along the shared dimension K. Each step multiplies one element of row `row` of A
    // with one element of column `col` of B, and adds it to the sum.
    for (int k = 0; k < K; ++k) {
      // A[row][k] is at index row*K + k. Lanes of one warp have different rows, so these
      // addresses are K floats apart. Each load touches a different 32-byte sector.
      // B[k][col] is at index k*N + col. All lanes of a warp have the same col, so every lane reads the
      // same address. The hardware serves that as a broadcast: one sector for all 32 lanes.
      acc += A[row * K + k] * B[k * N + col];
    }
    // Write the finished sum. Lanes write to C rows that are N floats apart: uncoalesced stores as well.
    C[row * N + col] = acc;
  }
}


// -----------------------------------------------------------------------------
// Host-side launcher: decides the grid and block shape, then launches the kernel
// -----------------------------------------------------------------------------
static void launch_naive(int M, int N, int K,
                         const float* A, const float* B,
                         float* C) {
  // 32 x 32 = 1024 threads per block, the maximum for most NVIDIA GPUs.
  // x runs over rows, y over columns (in this kernel's mapping).
  dim3 block(32, 32);
  // Enough blocks to cover the whole output. ceil_div rounds up for sizes that are not multiples of 32.
  dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
  // Launch on the default stream. The call returns immediately; the GPU runs the kernel later.
  sgemm_naive<<<grid, block>>>(M, N, K, A, B, C);
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
    const int M = 256, N = 256, K = 256;  // small enough for the CPU to finish in well under a second

    std::vector<float> hA((size_t)M * K),
                       hB((size_t)K * N),
                      ref((size_t)M * N);  // host (CPU) arrays
    fill_uniform(hA, 1);  // random inputs in [-1, 1), fixed seed 1
    fill_uniform(hB, 2);  // fixed seed 2

    DeviceBuffer<float> A(hA.size()),
                        B(hB.size()),
                        C((size_t)M * N);  // GPU arrays of the same sizes
    A.upload(hA);  // copy inputs to the GPU
    B.upload(hB);

    cpu_gemm(M, N, K, hA, hB, ref);  // reference result on the CPU

    launch_naive(M, N, K, A.ptr, B.ptr, C.ptr);  // run the GPU kernel

    CUDA_CHECK(cudaDeviceSynchronize());         // wait for it to finish, so runtime errors surface here

    std::vector<float> out = C.download();       // copy the result back to the CPU

    // Tolerance: K = 256 products of values in [-1, 1) accumulate small rounding differences.
    const bool ok = check_close("v0 naive vs CPU", ref.data(), out.data(), out.size(), 1e-3f, 1e-3f);

    if (!ok) return 1;  // a failed self-test makes the program exit with an error code
  }

  // ---- 2. Timing on a large problem.
  const int n = 2048;  // square problem: M = N = K = 2048

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
                                  launch_naive(n, n, n, A.ptr, B.ptr, C.ptr);
                               }, args.warmup, args.iters);

  // A matmul does 2*n^3 floating-point operations. Dividing by seconds and by 1e12 gives TFLOP/s.
  const double tflops = 2.0 * n * n * (double)n / (ms * 1e-3) / 1e12;
  printf("\n%-24s %10.3f ms  %8.4f TFLOP/s\n", "v0 naive", ms, tflops);

  // Optionally append a CSV row (--csv file). An empty path makes this a no-op.
  CsvWriter csv(args.csv);
  csv.row("matmul", "v0_naive", "2048x2048x2048", ms, tflops, "TFLOP/s");
  return 0;
}
