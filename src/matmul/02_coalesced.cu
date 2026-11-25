// =============================================================================
// matmul/02_coalesced.cu
// Matrix multiply, version 1: coalesced loads (same math, threadIdx.x walks columns)
// -----------------------------------------------------------------------------
// Problem:   C[M,N] = A[M,K] * B[K,N], all fp32 and row-major (as in 01_naive.cu).
// Idea:      The same algorithm as 01: one thread per output element, and each thread walks k and adds
//            A[row][k] * B[k][col]. Only the mapping from thread index to element changes. threadIdx.x
//            now picks the COLUMN and threadIdx.y picks the ROW. The block is 32 x 32 threads and the
//            grid is ceil_div(N,32) x ceil_div(M,32).
// Warp:      The GPU runs threads in groups of 32, called warps. The 32 threads of a warp are its lanes,
//            numbered 0..31. With a 32 x 32 block, a warp is 32 consecutive values of threadIdx.x at one
//            threadIdx.y. So one warp covers 32 consecutive columns of one row of C.
// Block:     a group of threads that run on one SM (one of the GPU's multiprocessors) and can share
//            shared memory. The grid is all the blocks of one launch.
// Work:      2*M*N*K FLOPs. At 2048^3 that is 17.2 GFLOP.
//            Minimum DRAM traffic at 2048^3 in fp32: 34 MB read, 17 MB written.
// Changed vs previous version: 01_naive.cu. Only the thread-to-element mapping changes. Row and column
//            swap roles, and the grid is ceil_div(N,32) x ceil_div(M,32) to match. The arithmetic and
//            the loop over k are unchanged.
// Why faster: The memory system moves data in 32-byte sectors. A warp's load touches every sector that
//            holds one of its 32 addresses, so fewer sectors per request means less wasted traffic.
//            - In 01, the 32 lanes of a warp have 32 different rows, so the A addresses are K floats
//              apart. One request touches about 16.5 sectors, and it uses one float out of each.
//            - Here, the B loads of a warp are 32 consecutive floats (one 128-byte run = 4 full sectors).
//            - The A load is the same address in all 32 lanes, so it is one sector, served as a broadcast.
//            - The C store is also 32 consecutive floats, so it is coalesced too.
// Run:       ./bin/matmul_02_coalesced                    self-test, then timing
//            ./bin/matmul_02_coalesced --profile          one launch only, for ncu
// Refs:      [NV-COALESCE] (coalescing and sectors), [PMPP] chapter 6 (coalescing), [CUDA-GUIDE] (warps)
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
// Every thread runs this same code; they differ only in their threadIdx and blockIdx values.
__global__ void sgemm_coalesced(int M, int N, int K,             // C is M x N, the shared dimension is K
                                const float* A, const float* B,  // inputs in GPU global memory
                                float* C) {                      // output in GPU global memory
  // Column of C that this thread computes. threadIdx.x changes fastest inside a block, so the 32 lanes
  // of a warp get 32 consecutive columns.
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Row of C that this thread computes. A warp has one threadIdx.y value, so all its lanes share a row.
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  // The grid is rounded up to whole 32 x 32 blocks, so some threads lie outside C. They must do nothing.
  if (row < M && col < N) {
    float acc = 0.f;  // running dot product for C[row][col]. It lives in a register, private to this thread.

    // Walk along the shared dimension K. Each step multiplies one element of A's row with one element of
    // B's column, and adds the product to acc.
    for (int k = 0; k < K; ++k) {
      // A[row][k] is at index row*K + k. All lanes of a warp have the same row, so they all read the same
      // address. The hardware serves this one load as a broadcast, using one sector.
      // B[k][col] is at index k*N + col. The lanes of a warp have consecutive col values, so they read 32
      // consecutive floats: one 128-byte run, which is 4 full sectors.
      acc += A[row * K + k] * B[k * N + col];
    }
    // Store the finished sum. The lanes of a warp write 32 consecutive floats of one row of C: coalesced.
    C[row * N + col] = acc;
  }
}


// -----------------------------------------------------------------------------
// Host-side launcher: decides the grid and block shape, then launches the kernel
// -----------------------------------------------------------------------------
static void launch_coalesced(int M, int N, int K,
                             const float* A, const float* B,
                             float* C) {
  // 32 x 32 = 1024 threads per block, the usual per-block maximum. x runs over columns in this kernel.
  dim3 block(32, 32);
  // x covers the N columns and y covers the M rows. ceil_div rounds up for sizes that are not multiples of 32.
  // Example at 2048^3: 64 x 64 = 4096 blocks, 1024 threads each, one thread per element of C.
  dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
  // The launch is asynchronous: it returns at once and the GPU runs the kernel later.
  sgemm_coalesced<<<grid, block>>>(M, N, K, A, B, C);
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
    A.upload(hA);  // copy the inputs to the GPU
    B.upload(hB);

    cpu_gemm(M, N, K, hA, hB, ref);  // reference result on the CPU

    launch_coalesced(M, N, K, A.ptr, B.ptr, C.ptr);  // run the GPU kernel

    CUDA_CHECK(cudaDeviceSynchronize());             // wait for it to finish, so runtime errors surface here

    std::vector<float> out = C.download();           // copy the result back to the CPU

    // Tolerance: 1e-3 absolute and 1e-3 relative. K = 256 products of values in [-1, 1) differ only by rounding.
    const bool ok = check_close("v1 coalesced vs CPU", ref.data(), out.data(), out.size(), 1e-3f, 1e-3f);

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
                                  launch_coalesced(n, n, n, A.ptr, B.ptr, C.ptr);
                               }, args.warmup, args.iters);

  // A matmul does 2*n^3 floating-point operations. Dividing by seconds and by 1e12 gives TFLOP/s.
  const double tflops = 2.0 * n * n * (double)n / (ms * 1e-3) / 1e12;
  printf("\n%-24s %10.3f ms  %8.4f TFLOP/s\n", "v1 coalesced", ms, tflops);

  // Optionally append a CSV row (--csv file). An empty path makes this a no-op.
  CsvWriter csv(args.csv);
  csv.row("matmul", "v1_coalesced", "2048x2048x2048", ms, tflops, "TFLOP/s");
  return 0;
}
