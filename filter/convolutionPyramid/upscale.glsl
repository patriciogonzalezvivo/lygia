#include "../../math/absi.glsl"

/*
author: Patricio Gonzalez Vivo
description: upscale for function for convolution pyramid  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: <vec4> convolutionPyramidUpscale(<sampler2D> tex0, sampler2D tex1, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - CONVOLUTIONPYRAMID_H1: 1.0334, 0.6836, 0.1507
    - CONVOLUTIONPYRAMID_H2: 0.0270
    - CONVOLUTIONPYRAMID_G: 0.7753, 0.0312

license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef CONVOLUTIONPYRAMID_H1
#define CONVOLUTIONPYRAMID_H1 1.0334, 0.6836, 0.1507
#endif

#ifndef CONVOLUTIONPYRAMID_H2
#define CONVOLUTIONPYRAMID_H2 0.0270
#endif

#ifndef CONVOLUTIONPYRAMID_G
#define CONVOLUTIONPYRAMID_G 0.7753, 0.0312
#endif

#ifndef FNC_CONVOLUTIONPYRAMID_UPSCALE
#define FNC_CONVOLUTIONPYRAMID_UPSCALE
vec4 convolutionPyramidUpscale(sampler2D tex0, sampler2D tex1, vec2 st, vec2 pixel) {
    const vec3  h1 = vec3(CONVOLUTIONPYRAMID_H1);
    const float h2 = CONVOLUTIONPYRAMID_H2;
    const vec2  g  = vec2(CONVOLUTIONPYRAMID_G);

    vec4 color = vec4(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel;
            color += SAMPLER_FNC(tex0, uv) * g[ absi(dx) ] * g[ absi(dy) ];
        }
    }

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel * 2.;
            color += SAMPLER_FNC(tex1, uv) * h2 * h1[ absi(dx) ] * h1[ absi(dy) ];
        }
    }

    return color;
}
#endif