#include "make.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the cross function for float3 types, to match GLSL's behavior.
use: <float3> cross(<float3> a, <float3> b)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_CROSS
#define FNC_CROSS

inline __host__ __device__ float3 cross(const float3& a, const float3& b) { return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }

#endif
