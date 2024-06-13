/*
contributors: Patricio Gonzalez Vivo
description: bias high pass
use: <float> highPass(<float> value, <float> bias)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HIGHPASS
#define FNC_HIGHPASS
inline __host__ __device__ float highPass(float value, float bias) { return max(value - bias, 0.0f) / (1.0f - bias); }
#endif
