#include "../sample/derivative.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a RGB normal map into normal vectors
use: normalMap(<sampler2D> texture, <float2> st, <float2> pixel)
options:
    - NORMALMAP_Z: Steepness of z before normalization, defaults to .01
*/
#ifndef NORMALMAP_Z
#define NORMALMAP_Z .01
#endif

#ifndef FNC_NORMALMAP
#define FNC_NORMALMAP
float3 normalMap(sampler2D tex, float2 st, float2 pixel) {
    float2 deltas = sampleDerivative(tex, st, pixel);
    return normalize(float3(deltas.x, deltas.y, NORMALMAP_Z) );
}
#endif