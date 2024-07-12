/*
contributors: none
description: none
use: <float3> spectral_geoffrey(<float> x)
*/

#ifndef FNC_SPECTRAL_GEOFFREY
#define FNC_SPECTRAL_GEOFFREY
float3 spectral_geoffrey(float t) {
    float3 r = (t * 2.0 - 0.5) * 2.1 - float3(1.8, 1.14, 0.3);
    return 0.99 - r * r;
}
#endif