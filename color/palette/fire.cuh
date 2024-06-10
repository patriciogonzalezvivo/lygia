#include "../../math/operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: Simpler fire color ramp
use: <float3> fire(<float> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FIRE
#define FNC_FIRE

inline __host__ __device__  float3 fire(float x) { return make_float3(1.0f, 0.25f, 0.0625f) * exp(4.0 * x - 1.0); }

#endif