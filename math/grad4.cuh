#include "make.cuh"
#include "abs.cuh"
#include "dot.cuh"
#include "frac.cuh"
#include "floor.cuh"
#include "step.cuh"
#include "operations.cuh"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: grad4, used for snoise(float4 v)
use: grad4(<float> j, <float4> ip)
*/

#ifndef FNC_GRAD4
#define FNC_GRAD4

inline __host__ __device__ float4 grad4(float j, float4 ip) {
    float4 p, s;

    p.x = floor( frac(j * ip.x) * 7.0f) * ip.z - 1.0f;
    p.y = floor( frac(j * ip.y) * 7.0f) * ip.z - 1.0f;
    p.z = floor( frac(j * ip.z) * 7.0f) * ip.z - 1.0f;

    p.w = 1.5f - dot( abs( make_float3(p.x, p.y, p.z) ), make_float3(1.0f) );
    
    // GLSL: s = float4(lessThan(p, float4(0.0)));
    s = 1.0f - step(make_float4(0.0f), p);

    p.x = p.x + (s.x * 2.0f - 1.0f) * s.w;
    p.y = p.y + (s.y * 2.0f - 1.0f) * s.w;
    p.z = p.z + (s.z * 2.0f - 1.0f) * s.w;

    return p;
}

#endif
