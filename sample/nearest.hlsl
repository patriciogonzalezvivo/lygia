#include "lygia/math/nearest.hlsl"

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

#ifndef FNC_SAMPLENEARERS
#define FNC_SAMPLENEARERS
float4 sampleNearest(sampler2D tex, float2 st, float2 texResolution) {
    return SAMPLER_FNC( tex, nearest(st, texResolution) );
}
#endif