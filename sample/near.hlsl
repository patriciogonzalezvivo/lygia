#include "lygia/math/decimation.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: fakes a nearest sample
use:> <float4 sampleNear(<sampler2D> tex, <float2> st, <float2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) tex2D(TEX, UV)
#endif

#ifndef FNC_SAMPLENEAR
#define FNC_SAMPLENEAR
float4 sampleNear(sampler2D tex, float2 st, float2 texResolution) {
    float2 half_pixel = 0.5/texResolution;
    return SAMPLER_FNC( tex, decimation(st, texResolution) + half_pixel );
}
#endif