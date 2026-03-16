#include "derivative.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB normal map into normal vectors
use: SAMPLEBUMPMap(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel)
options:
    - SAMPLEBUMPMAP_Z: Steepness of z before normalization, defaults to .01
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/
// #define SAMPLEBUMPMAP_Z .01

fn sampleBumpMap(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f) -> vec3f {
    let deltas = sampleDerivative(tex, st, pixel);
    return normalize( vec3f(deltas, SAMPLEBUMPMAP_Z) );
}
