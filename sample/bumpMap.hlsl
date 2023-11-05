#include "derivative.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB normal map into normal vectors
use: SAMPLEBUMPMap(<SAMPLER_TYPE> texture, <float2> st, <float2> pixel)
options:
    - SAMPLEBUMPMAP_Z: Steepness of z before normalization, defaults to .01
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