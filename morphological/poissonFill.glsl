// POISSON FILL (DEFAULT)
// #define POISSONFILL_H1 1.0334, 0.6836, 0.1507
// #define POISSONFILL_H2 0.0270
// #define POISSONFILL_G 0.7753, 0.0312

// LAPLACIAN INTEGRATOR
// #define POISSONFILL_H1 0.7, 0.5, 0.15
// #define POISSONFILL_H2 1.0
// #define POISSONFILL_G  0.547, 0.175

#include "poissonFill/downscale.glsl"
#include "poissonFill/upscale.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: down and up scaling functions for convolution pyramid  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: convolutionPyramid(<sampler2D> texture0, <sampler2D> texture1, <vec2> st, <vec2> pixel, <bool> upscale)
options:
    - POISSONFILL_H1: 1.0334, 0.6836, 0.1507
    - POISSONFILL_H2: 0.0270
    - POISSONFILL_G: 0.7753, 0.0312
*/

#ifndef FNC_POISSONFILL
#define FNC_POISSONFILL
vec4 poissonFill(sampler2D tex0, sampler2D tex1, vec2 st, vec2 pixel, bool upscale) {
    vec4 color = vec4(0.0);
    if (!upscale) {
        color = poissonFillDownscale(tex0, st, pixel);
    }
    else {
        color = poissonFillUpscale(tex0, tex1, st, pixel);
    }
    return (color.a == 0.0)? color : vec4(color.rgb/color.a, 1.0);
}
#endif