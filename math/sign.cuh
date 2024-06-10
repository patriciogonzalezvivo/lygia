#include <cuda_runtime.h>

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the sign function for float types, to match GLSL's behavior.
use: <float> sign(<float> value);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SIGN
#define FNC_SIGN
inline __host__ __device__ float sign(float _v) { return _v > 0.0 ? 1.0f : -1.0f; }
#endif