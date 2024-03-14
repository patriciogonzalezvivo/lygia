#include <cuda_runtime.h>

#ifndef FNC_SIGN
#define FNC_SIGN
inline __host__ __device__ float sign(float _v) { return _v > 0.0 ? 1.0f : -1.0f; }
#endif