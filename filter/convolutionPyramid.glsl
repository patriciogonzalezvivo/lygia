/*
author: Patricio Gonzalez Vivo
description: down and up scaling functions for convolution pyramid  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: convolutionPyramid(<sampler2D> texture0, <sampler2D> texture1, <vec2> st, <vec2> pixel, <bool> upscale)
options:
    - CONVOLUTIONPYRAMID_H1: 1.0334, 0.6836, 0.1507
    - CONVOLUTIONPYRAMID_H2: 0.0270
    - CONVOLUTIONPYRAMID_G: 0.7753, 0.0312

license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

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