#include "../sample.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: turns alpha to zero if it's outside the texture normalize coordinates
use: <float4> sampleZero(<SAMPLER_TYPE> tex, <float2> st);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef FNC_SAMPLEZERO
#define FNC_SAMPLEZERO
float4 sampleZero(SAMPLER_TYPE tex, float2 st) { 
    return SAMPLER_FNC( tex, st ) * float4(1.0,1.0,1.0, (st.x <= 0.0 || st.x >= 1.0 || st.y <= 0.0 || st.y >= 1.0)? 0.0 : 1.0); 
}
#endif