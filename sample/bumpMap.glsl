#include "derivative.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB normal map into normal vectors
use: SAMPLEBUMPMap(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel)
options:
    - SAMPLEBUMPMAP_Z: Steepness of z before normalization, defaults to .01
*/
#ifndef SAMPLEBUMPMAP_Z
#define SAMPLEBUMPMAP_Z .01
#endif

#ifndef FNC_SAMPLEBUMPMAP
#define FNC_SAMPLEBUMPMAP
vec3 sampleBumpMap(SAMPLER_TYPE tex, vec2 st, vec2 pixel) {
    vec2 deltas = sampleDerivative(tex, st, pixel);
    return normalize( vec3(deltas, SAMPLEBUMPMAP_Z) );
}
#endif