// POISSON FILL (DEFAULT)
// #define PYRAMID_H1 1.0334, 0.6836, 0.1507
// #define PYRAMID_H2 0.0270
// #define PYRAMID_G 0.7753, 0.0312

// LAPLACIAN INTEGRATOR
// #define PYRAMID_H1 0.7, 0.5, 0.15
// #define PYRAMID_H2 1.0
// #define PYRAMID_G  0.547, 0.175

#include "pyramid/downscale.glsl"
#include "pyramid/upscale.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: down and up scaling functions for convolution pyramid  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: pyramid(<SAMPLER_TYPE> texture0, <SAMPLER_TYPE> texture1, <vec2> st, <vec2> pixel,
  <bool> upscale)
options:
    - PYRAMID_H1: 1.0334, 0.6836, 0.1507
    - PYRAMID_H2: 0.027
    - PYRAMID_G: 0.7753, 0.0312
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_PYRAMID
#define FNC_PYRAMID
vec4 pyramid(SAMPLER_TYPE tex0, SAMPLER_TYPE tex1, vec2 st, vec2 pixel, bool upscale) {
    vec4 color = vec4(0.0);
    if (!upscale) {
        color = pyramidDownscale(tex0, st, pixel);
    }
    else {
        color = pyramidUpscale(tex0, tex1, st, pixel);
    }
    return (color.a == 0.0)? color : vec4(color.rgb/color.a, 1.0);
}
#endif