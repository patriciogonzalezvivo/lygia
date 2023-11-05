#include "../../math/operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: heatmap palette
use: <float3> heatmap(<float> value)
*/

#ifndef FNC_HEATMAP
#define FNC_HEATMAP

inline __host__ __device__ float3 heatmap(float v) {
    float3 r = v * 2.1 - make_float3(1.8f, 1.14f, 0.3f);
    return 1.0 - r * r;
}

#endif