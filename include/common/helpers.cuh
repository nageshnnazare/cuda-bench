// =============================================================================
// common/helpers.cuh
// -----------------------------------------------------------------------------
// Every program can includes this one file. 
// It provides:
//   CUDA_CHECK, CUDA_CHECK_LAST  turn CUDA errors into messages with file and line
//   ceil_div                     integer division rounded up (used to size grids)
//   BenchArgs                    command-line flags: --profile, --iters, --warmup, --csv
//   time_ms                      average kernel time, measured with CUDA events
//   fill_uniform, round_to_half, to_half   random test data and fp16 conversion
//   check_close                  element-wise comparison with an absolute and relative tolerance
//   DeviceBuffer<T>              RAII wrapper around cudaMalloc / cudaFree
//   memcpy_roofline_gbps         the bandwidth ceiling for memory-bound kernels
//   print_device, CsvWriter      printing the GPU and appending CSV rows
// =============================================================================

#ifndef _COMMON_HELPERS_
#define _COMMON_HELPERS_

#include <cuda_fp16.h>      // the __half type and __float2half / __half2float conversions
#include <cuda_runtime.h>   // the CUDA runtime API: cudaMalloc, cudaMemcpy, events, ...
#include <algorithm>        // std::max, std::min
#include <cmath>            // std::fabs, std::isnan, std::sqrt, INFINITY
#include <cstdint>          // uint32_t
#include <cstdio>           // printf, fprintf, fopen, fclose
#include <cstdlib>          // std::exit, std::atoi
#include <cstring>          // strcmp
#include <random>           // std::mt19937 and std::uniform_real_distribution
#include <string>           // std::string
#include <vector>           // std::vector


// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

// CUDA_CHECK(call) runs one CUDA runtime call and stops the program if it failed.
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)


// CUDA_CHECK_LAST() checks the error state after a kernel launch.
// Kernel launches are asynchronous and return no status. cudaGetLastError() reports
// launch-configuration errors (too many threads per block, too much shared memory, ...).
// It does not report errors raised while the kernel is running. Those show up at the
// next call that synchronises with the GPU, such as cudaDeviceSynchronize or a cudaMemcpy.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())


// Integer division rounded up: ceil_div(10, 4) == 3, ceil_div(8, 4) == 2.
// Used to compute how many blocks are needed so that every element gets a thread.
// Function can be called from CPU code and GPU kernels.
__host__ __device__ inline constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }


// ---------------------------------------------------------------------------
// Command-line arguments shared by every learning program
// ---------------------------------------------------------------------------

struct BenchArgs {
  bool profile = false;  // --profile: run exactly one launch of each kernel and skip the self-test
  int warmup = 10;       // number of untimed launches before the timed ones (warms caches and clocks)
  int iters = 50;        // number of timed launches; the reported time is their average
  std::string csv;       // --csv <path>: append one result row to this file (empty means no CSV)

  // Parse argv into a BenchArgs. Unknown flags are ignored.
  static BenchArgs parse(int argc, char** argv) {
    BenchArgs a;                            // start from the defaults declared above
    for (int i = 1; i < argc; ++i) {        // argv[0] is the program name, so start at 1
      if (!strcmp(argv[i], "--profile")) {  // strcmp returns 0 when the two strings are equal
        a.profile = true;
        a.warmup = 0;                       // no warm-up: ncu should see the first and only launch
        a.iters = 1;                        // exactly one timed launch
      } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
        a.iters = std::atoi(argv[++i]);     // ++i advances to the value that follows the flag
      } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
        a.warmup = std::atoi(argv[++i]);
      } else if (!strcmp(argv[i], "--csv") && i + 1 < argc) {
        a.csv = argv[++i];
      }
    }
    return a;
  }
};


// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

// Average time of one call to f(), in milliseconds, measured with CUDA events.
// f is any callable, usually a lambda that launches a kernel. The template lets the
// compiler inline the lambda into the timing loop.
template <typename F>
float time_ms(F&& f, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) f();     // warm-up: first launches pay one-time costs (module load, cold caches)
 
  CUDA_CHECK(cudaDeviceSynchronize());      // wait until all warm-up work has finished on the GPU

  cudaEvent_t start, stop;                  // events are timestamps recorded by the GPU itself
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));       // timestamp is queued in the stream just before the work

  for (int i = 0; i < iters; ++i) f();      // the timed launches; they are queued back to back

  CUDA_CHECK(cudaEventRecord(stop));        // timestamp is queued just after the last launch
  CUDA_CHECK(cudaEventSynchronize(stop));   // block the CPU until the GPU has reached `stop`
  CUDA_CHECK_LAST();                        // report any launch error that happened during timing

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));  // GPU time between the two events, in milliseconds
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return ms / iters;                        // average time for one call
}


// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

// Fill v with uniform random floats in [lo, hi). The fixed seed makes every run produce the same data.
inline void fill_uniform(std::vector<float>& v, uint32_t seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);                                 // Mersenne Twister pseudo-random generator
  std::uniform_real_distribution<float> dist(lo, hi);     // maps raw generator output into [lo, hi)
  for (auto& x : v) x = dist(rng);                        // each element is overwritten in place
}


// Round every value to fp16 and back to fp32. The CPU reference then sees exactly the values
// the GPU sees, so the comparison only has to allow for different summation order.
inline std::vector<float> round_to_half(const std::vector<float>& v) {
  std::vector<float> out(v.size());  // same length as the input
  for (size_t i = 0; i < v.size(); ++i) out[i] = __half2float(__float2half(v[i]));  // fp32 -> fp16 (rounds) -> fp32
  return out;
}


// Convert fp32 values to fp16 storage, which is what the tensor-core kernels read.
inline std::vector<half> to_half(const std::vector<float>& v) {
  std::vector<half> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __float2half(v[i]);  // rounds to nearest fp16
  return out;
}


// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------

// Returns true when |out - ref| <= atol + rtol * |ref| for every element.
// atol is the absolute tolerance and rtol the relative one; both are needed because
// values near zero need absolute slack and large values need relative slack.
inline bool check_close(const char* name, const float* ref, const float* out,
                        size_t n, float atol, float rtol,
                        bool verbose = true) {
  double max_abs = 0.0;     // largest absolute error seen
  size_t bad = 0;           // number of elements that fail the tolerance
  size_t first_bad = 0;     // index of the first failing element, for the message
  for (size_t i = 0; i < n; ++i) {
    const double diff = std::fabs((double)out[i] - (double)ref[i]);  // absolute error for this element
    // Written as "not (diff <= limit)" on purpose: a NaN diff makes the comparison false,
    // so NaN values are counted as failures.
    if (!(diff <= atol + rtol * std::fabs((double)ref[i]))) {
      if (bad == 0) first_bad = i;  // remember where the first failure is
      ++bad;
    }
    // Track the worst error. A NaN diff becomes INFINITY so it is reported clearly.
    if (diff > max_abs || std::isnan(diff)) max_abs = std::isnan(diff) ? INFINITY : diff;
  }
  if (verbose || bad) {
    // One status line per check: "ok" or "FAIL", the name, and the worst absolute error.
    printf("  [%s] %-28s max_abs_err=%.3e",
                              bad ? "FAIL" : " ok ", name, max_abs);
    if (bad)  // on failure also print the first offending value next to the expected one
      printf("  (%zu bad, first at %zu: got %f want %f)", 
                              bad, first_bad, out[first_bad], ref[first_bad]);
    printf("\n");
  }
  return bad == 0;  // true means every element passed
}


// ---------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------

// A typed array in GPU global memory. The destructor frees it, so it cannot leak.
// The copy constructor and copy assignment are deleted: two objects must not free the same pointer.
template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;  // device pointer (valid only on the GPU; do not dereference on the CPU)
  size_t n = 0;      // number of T elements

  DeviceBuffer() = default;
  // Allocate `count` elements of T on the GPU. cudaMalloc does not zero the memory.
  explicit DeviceBuffer(size_t count) : n(count) { CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T))); }
  DeviceBuffer(const DeviceBuffer&) = delete;             // copying would double-free
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;  // same reason
  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);  // release GPU memory when the object goes out of scope
  }

  // Copy a host vector into this buffer (host to device).
  void upload(const std::vector<T>& h) {
    CUDA_CHECK(cudaMemcpy(ptr, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }

  // Copy this buffer back to a new host vector (device to host). cudaMemcpy waits for prior kernels.
  std::vector<T> download() const {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }

  // Set every byte to zero (useful before a kernel that accumulates into the output).
  void zero() { CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T))); }
};


// ---------------------------------------------------------------------------
// Roofline reference for memory-bound kernels
// ---------------------------------------------------------------------------

// The bandwidth ceiling for a kernel that reads `bytes` and writes `bytes`: a device-to-device
// memcpy of the same size. A memcpy does exactly that traffic and no arithmetic, so it shows
// the fastest this GPU can move that many bytes. Returns GB/s.
inline double memcpy_roofline_gbps(size_t bytes, int warmup, int iters) {
  DeviceBuffer<char> src(bytes), dst(bytes);  // two buffers on the GPU, each `bytes` long
  float ms = time_ms( [&] {
                            CUDA_CHECK(cudaMemcpyAsync(dst.ptr, src.ptr, bytes, cudaMemcpyDeviceToDevice)); 
                          }, warmup, iters);
  return 2.0 * bytes / (ms * 1e-3) / 1e9;  // 2x because bytes are read AND written; ms -> s; bytes/s -> GB/s
}


// ---------------------------------------------------------------------------
// Printing and CSV output
// ---------------------------------------------------------------------------

// Print the name, compute capability (sm_XY), SM count and memory size of the current GPU.
inline void print_device() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));              // which GPU this process is using (0 by default)
  cudaDeviceProp p{};                           // zero-initialised property struct
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev)); // fill it in for that GPU
  printf("Device: %s (sm_%d%d, %d SMs, %.1f GB)\n", p.name, p.major, p.minor, p.multiProcessorCount,
         p.totalGlobalMem / 1e9);
}


// Appends one result row per call to a CSV file. Columns: op,version,shape,ms,metric,unit.
struct CsvWriter {
  FILE* f = nullptr;  // nullptr means "do not write"

  explicit CsvWriter(const std::string& path) {
    if (path.empty()) return;                  // no --csv given: stay silent
    FILE* probe = fopen(path.c_str(), "r");    // does the file already exist?
    const bool exists = probe != nullptr;
    if (probe) fclose(probe);
    f = fopen(path.c_str(), "a");              // open for appending
    if (f && !exists) fprintf(f, "op,version,shape,ms,metric,unit\n");  // write the header once
  }

  ~CsvWriter() {
    if (f) fclose(f);  // close the file when the writer goes out of scope
  }

  // Write one row. `metric` is the derived number, e.g. TFLOP/s or GB/s, and `unit` names it.
  void row(const char* op, const char* version, const std::string& shape, double ms, double metric,
           const char* unit) {
    if (f) fprintf(f, "%s,%s,%s,%.5f,%.3f,%s\n", op, version, shape.c_str(), ms, metric, unit);
  }
};

#endif //_COMMON_HELPERS_
