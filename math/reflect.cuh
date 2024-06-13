#include "dot.cuh"
#include "operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the reflect function which returns the reflection of a vector to match GLSL's language
use: float3 reflect(float3 i, float3 n)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


#ifndef FNC_REFLECT
#define FNC_REFLECT

inline __host__ __device__ float3 reflect(const float3& i, const float3& n) { return i - 2.0f * n * dot(n, i); }

#endif