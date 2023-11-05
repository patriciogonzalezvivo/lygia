#include "../math/dot.cuh"
#include "../math/length.cuh"

/*
contributors:  Inigo Quiles
description: generate the SDF of a plane
use: <float> planeSDF( in <float3> pos, in <vec2> h ) 
*/

#ifndef FNC_PLANESDF
#define FNC_PLANESDF

inline  __host__ __device__ float planeSDF(float3 p) {  return p.y; }
inline  __host__ __device__ float planeSDF(float3 p, float3 planePoint, float3 planeNormal) { return (dot(planeNormal, p) + dot(planeNormal, planePoint)) / length(planeNormal); }

#endif