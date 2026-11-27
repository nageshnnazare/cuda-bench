// =============================================================================
// matmul/03_smem_tiling.cu
// Matrix multiply, version 2: shared-memory tiling with 32 x 32 tiles
// -----------------------------------------------------------------------------
// Problem:   C[M,N] = A[M,K] * B[K,N], all fp32 and row-major. M, N and K can be any size; the
//            edges are handled by zero padding (see the kernel).
// Idea:      A block of 32 x 32 threads computes one 32 x 32 tile of C. It walks along K in steps of 32.
//            At each step every thread copies one element of the current A tile and one element of the
//            current B tile from global memory into shared memory. Shared memory is on-chip memory that
//            all threads of one block can see. A barrier makes the copies visible to the whole block.
//            Then each thread computes 32 products from shared memory, and a second barrier protects the
//            tiles before they are overwritten.
// Work:      2*M*N*K FLOPs. At 2048^3 that is 17.2 GFLOP.
//            Global loads: each block loads every element of its A strip and B strip once per tile,
//            and the 32 threads that need an element read it from shared memory instead of from global
//            memory.
// Changed vs previous version: 02_coalesced.cu. The mapping and the grid are the same. Added: the loop over K
//            in steps of TILE, two shared-memory tiles As and Bs (4 KB each), zero padding at the edges,
//            and two __syncthreads() barriers per step.
// Why faster: Shared memory is on-chip, far faster than global memory. Each element of A and B is loaded
//            from global memory once per block, not once per multiply-add. This cuts the L1/L2 requests
//            by about 32x. Shared memory is split into 32 banks, each 4 bytes wide. Threads of one
//            warp that touch different words in the same bank are served one after another (a bank
//            conflict). The loads below are conflict-free, as the comments in the kernel explain.
//
//            Why the DRAM bytes barely change each block reads a BM-row strip of A (BM x K) and a BN-column 
//            strip of B (K x BN), and there are (M/BM) * (N/BN) blocks. So the DRAM reads come to about
//                M * N * K * bytes * (1/BM + 1/BN)
//            With BM = BN = 32 at 2048^3 in fp32 that is 2048^3 * 4 * (1/32 + 1/32), about 2.15 GB. 
//            The tile size sets this term, not the use of shared memory. Doubling the tile halves it: 
//            64 x 64 gives about 1,070 MB, and 128 x 128 about 537 MB. The 4 MB L2 cache catches some 
//            re-reads, so the measured numbers are lower (see below).
// Run:       ./bin/matmul_03_smem_tiling                    self-test, then timing
//            ./bin/matmul_03_smem_tiling --profile          one launch only, for ncu
// Refs:      [NV-SMEM] (shared memory and bank conflicts), [PMPP] chapter 5 (tiling), [BOEHM] (the shared-memory step),
//            [CUDA-GUIDE] (__syncthreads and shared memory)
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
// TILE is a template parameter, so it is a compile-time constant. A block has TILE x TILE threads and
// computes one TILE x TILE tile of C.
template <int TILE>
__global__ void sgemm_smem(int M, int N, int K,             // C is M x N, the shared dimension is K
                           const float* A, const float* B,  // inputs in GPU global memory
                           float* C) {                      // output in GPU global memory
  // __shared__ puts an array in the block's shared memory: one copy per block, visible to all of its
  // threads, and much faster than global memory. As holds a TILE x TILE piece of A and Bs a TILE x TILE
  // piece of B. With TILE = 32 each is 32 * 32 * 4 bytes = 4 KB.
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  // (tx, ty) is this thread's position inside its block. tx is the column inside the tile, ty the row.
  const int tx = threadIdx.x, ty = threadIdx.y;
  // Global position of this thread's output element. blockIdx.y picks the tile row and blockIdx.x the tile
  // column, and the thread's position inside the block adds the offset within the tile.
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  float acc = 0.f;  // this thread's output. It stays in a register across all the K tiles.

  // Walk along K one tile at a time. k0 is the first k of the current tile.
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // ---- Load step. Each thread loads ONE element of As and ONE of Bs.
    // The conditional expressions implement zero padding. An element outside the matrix (row >= M, or
    // k0 + tx >= K for A, or k0 + ty >= K or col >= N for B) is replaced by 0.0. Multiplying by 0 adds
    // nothing to acc, so the padded products do not change the answer.
    // A[row][k0 + tx] is at index row*K + k0 + tx. Threads with the same ty share a row, and their tx values
    // are consecutive, so one warp reads 32 consecutive floats of A: coalesced.
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + (k0 + tx)] : 0.f;
    // B[k0 + ty][col] is at index (k0 + ty)*N + col. Consecutive tx gives consecutive col, so this is also coalesced.
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;

    // First barrier. Every thread must finish its load before anyone reads As or Bs. A thread reads
    // elements that other threads loaded, so without this barrier it could read a value that is not
    // written yet. __syncthreads() waits until all threads of the block reach it, and makes their shared
    // memory writes visible to the block.
    __syncthreads();

    // #pragma unroll asks the compiler to write out all TILE iterations of the loop below, with no loop
    // counter or branch. The k in As[ty][k] and Bs[k][tx] then becomes a fixed offset.
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      // As[ty][k]: every lane of a warp has the same ty, so all lanes read the same address. This is a
      // broadcast and causes no bank conflict.
      // Bs[k][tx]: consecutive tx gives 32 consecutive floats, one per bank. This is conflict-free.
      acc += As[ty][k] * Bs[k][tx];  // one multiply-add of this thread's dot product
    }

    // Second barrier. Nobody may start the next tile's loads, which overwrite As and Bs, until every
    // thread has finished reading the current tile.
    __syncthreads();
  }

  // Only threads that own a real element store it. Every thread still reached both barriers in each step,
  // which is why the kernel never returns early.
  if (row < M && col < N) C[row * N + col] = acc;
}


// -----------------------------------------------------------------------------
// Host-side launcher: decides the grid and block shape, then launches the kernel
// -----------------------------------------------------------------------------
static void launch_smem(int M, int N, int K, const float* A, const float* B, float* C) {
  constexpr int TILE = 32;  // tile edge. A block is TILE x TILE threads: one thread per element of its C tile
  dim3 block(TILE, TILE);   // 32 x 32 = 1024 threads, the usual per-block maximum
  // One block for each TILE x TILE tile of C. x runs over the N columns and y over the M rows. ceil_div rounds
  // up, so sizes that are not multiples of 32 get a partial tile at the edge.
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  sgemm_smem<TILE><<<grid, block>>>(M, N, K, A, B, C);  // the template argument fixes TILE at compile time
  CUDA_CHECK_LAST();                                    // catch an invalid launch configuration right away
}


// -----------------------------------------------------------------------------
// main: self-test on a small problem, then timing on a large one
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
  BenchArgs args = BenchArgs::parse(argc, argv);  // read --profile, --iters, --warmup, --csv
  print_device();                                 // show which GPU is being used

  // ---- 1. Self-test: compare against the CPU. Skipped under --profile so ncu sees exactly one launch.
  if (!args.profile) {
    // 256 is a multiple of 32, so this test never takes the zero-padding branches. Those branches matter
    // only when M, N or K is not a multiple of 32.
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

    launch_smem(M, N, K, A.ptr, B.ptr, C.ptr);  // run the GPU kernel

    CUDA_CHECK(cudaDeviceSynchronize());        // wait for it to finish, so runtime errors surface here

    std::vector<float> out = C.download();      // copy the result back to the CPU

    // Tolerance: 1e-3 absolute and 1e-3 relative. K = 256 products of values in [-1, 1) differ only by rounding.
    const bool ok = check_close("v2 smem tiling vs CPU", ref.data(), out.data(), out.size(), 1e-3f, 1e-3f);

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
                                  launch_smem(n, n, n, A.ptr, B.ptr, C.ptr); 
                               }, args.warmup, args.iters);

  // A matmul does 2*n^3 floating-point operations. Dividing by seconds and by 1e12 gives TFLOP/s.
  const double tflops = 2.0 * n * n * (double)n / (ms * 1e-3) / 1e12;
  printf("\n%-24s %10.3f ms  %8.4f TFLOP/s\n", "v2 smem tiling", ms, tflops);

  // Optionally append a CSV row (--csv file). An empty path makes this a no-op.
  CsvWriter csv(args.csv);
  csv.row("matmul", "v2_smem_tiling", "2048x2048x2048", ms, tflops, "TFLOP/s");
  return 0;
}

