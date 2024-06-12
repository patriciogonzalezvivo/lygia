#include "../math/cubic.hlsl"
#include "../math/quintic.hlsl"
#include "../sampler.hlsl"

/*
contributors: Inigo Quiles
description: avoid the ugly artifacts of bilinear texture filtering. You can find more information here https://iquilezles.org/articles/texture
use: <float4 sampleSmooth(<SAMPLER_TYPE> tex, <float2> st, <float2> texResolution)
options:
    - SAMPLER_FNC(TEX, UV): nan
    - SAMPLESMOOTH_POLYNOMIAL: cubic or quartic
*/

#ifndef SAMPLESMOOTH_POLYNOMIAL
#define SAMPLESMOOTH_POLYNOMIAL cubic
#endif

#ifndef FNC_SAMPLESMOOTH
#define FNC_SAMPLESMOOTH
float4 sampleSmooth(SAMPLER_TYPE tex, float2 st, float2 texResolution) {
    st *= texResolution + 0.5;
    float2 fst = frac( st );
    st = floor( st );
    st += SAMPLESMOOTH_POLYNOMIAL(fst);
    st = (st - 0.5) / texResolution;
    return SAMPLER_FNC( tex, st );
}
#endif