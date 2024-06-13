#include "../../math/abs.cuh"
#include "../../math/mod.cuh"
#include "../../math/smoothstep.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: 'Physical Hue. Ratio: 1/3 = neon, 1/4 = refracted, 1/5+ = approximate white'
use: <float3> hue(<float> hue[, <float> ratio])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_PALETTE_HUE
#define FNC_PALETTE_HUE

inline __host__ __device__ float3 hue(float _hue, float _ratio) {
    return smoothstep(  make_float3(0.0f), make_float3(1.0f), 
                        abs( mod( _hue + make_float3(0.0f, 1.0f, 2.0f) * _ratio, 1.0f) * 2.0f - 1.0f));
}

inline __host__ __device__ float3 hue(float _hue) { return hue(_hue, 0.33333f); }

#endif