#include "../../math/saturate.cuh"
#include "../../math/pow.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: Simpler water color ramp
use: <float3> water(<float> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_WATER
#define FNC_WATER

inline __host__ __device__ float3 water(float x) {
    x = 4.0f * saturate(1.0f - x);
    return pow( make_float3(0.1f, 0.7f, 0.8f), make_float3(x));
}

#endif