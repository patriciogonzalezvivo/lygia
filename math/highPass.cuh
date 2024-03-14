/*
contributors: Patricio Gonzalez Vivo
description: bias high pass
use: <float> highPass(<float> value, <float> bias)
*/

#ifndef FNC_HIGHPASS
#define FNC_HIGHPASS
inline __host__ __device__ float highPass(float value, float bias) { return max(value - bias, 0.0f) / (1.0f - bias); }
#endif
