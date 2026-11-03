#include <hip/hip_runtime.h>
// Mimic top-k launch: grid=rows blocks x 512 threads; each thread loads one logit then atomicAdd on a shared counter.
// Record per (block, wave): arrival rank, and per block the XCC id (HW_REG_XCC_ID = hwreg 20).
__global__ void k(const float* x, int* rank, int* xcc, int n) {
  __shared__ int cnt;
  if (threadIdx.x == 0) cnt = 0;
  __syncthreads();
  float v = x[(blockIdx.x * 384 + threadIdx.x * 7) % n];
  int slot = -1;
  if (v > -1e30f) slot = atomicAdd(&cnt, 1);
  if ((threadIdx.x & 63) == 0) rank[blockIdx.x * 8 + threadIdx.x / 64] = slot;   // lane0's slot of each wave
  if (threadIdx.x == 0) xcc[blockIdx.x] = __builtin_amdgcn_s_getreg(20 | (0 << 6) | (3 << 11)); // HW_REG_XCC_ID, 4 bits
}
extern "C" void launch(const float* x, int* rank, int* xcc, int n, int rows) { hipLaunchKernelGGL(k, dim3(rows), dim3(512), 0, 0, x, rank, xcc, n); }
