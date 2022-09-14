// POISSON FILL (DEFAULT)
// #define CONVOLUTIONPYRAMID_H1 1.0334, 0.6836, 0.1507
// #define CONVOLUTIONPYRAMID_H2 0.0270
// #define CONVOLUTIONPYRAMID_G 0.7753, 0.0312

// LAPLACIAN INTEGRATOR
// #define CONVOLUTIONPYRAMID_H1 0.7, 0.5, 0.15
// #define CONVOLUTIONPYRAMID_H2 1.0
// #define CONVOLUTIONPYRAMID_G  0.547, 0.175

#include "convolutionPyramid/downscale.glsl"
#include "convolutionPyramid/upscale.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: down and up scaling functions for convolution pyramid  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: convolutionPyramid(<sampler2D> texture0, <sampler2D> texture1, <vec2> st, <vec2> pixel, <bool> upscale)
options:
    - CONVOLUTIONPYRAMID_H1: 1.0334, 0.6836, 0.1507
    - CONVOLUTIONPYRAMID_H2: 0.0270
    - CONVOLUTIONPYRAMID_G: 0.7753, 0.0312
*/

#ifndef FNC_CONVOLUTIONPYRAMID
#define FNC_CONVOLUTIONPYRAMID
vec4 convolutionPyramid(sampler2D tex0, sampler2D tex1, vec2 st, vec2 pixel, bool upscale) {
    vec4 color = vec4(0.0);
    if (!upscale) {
        color = convolutionPyramidDownscale(tex0, st, pixel);
    }
    else {
        color = convolutionPyramidUpscale(tex0, tex1, st, pixel);
    }
    return (color.a == 0.0)? color : vec4(color.rgb/color.a, 1.0)
}
#endif