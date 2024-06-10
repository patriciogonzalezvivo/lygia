#include "derivative.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB normal map into normal vectors
use: SAMPLEBUMPMap(<SAMPLER_TYPE> texture, <float2> st, <float2> pixel)
options:
    - SAMPLEBUMPMAP_Z: Steepness of z before normalization, defaults to .01
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/
#ifndef SAMPLEBUMPMAP_Z
#define SAMPLEBUMPMAP_Z .01
#endif

#ifndef FNC_SAMPLEBUMPMAP
#define FNC_SAMPLEBUMPMAP
float3 sampleBumpMap(SAMPLER_TYPE tex, float2 st, float2 pixel) {
    float2 deltas = sampleDerivative(tex, st, pixel);
    return normalize(float3(deltas.x, deltas.y, SAMPLEBUMPMAP_Z) );
}
#endif