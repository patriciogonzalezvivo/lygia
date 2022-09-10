#include "../../math/absi.glsl"

/*
author: Patricio Gonzalez Vivo
description: downscale for function for convolution pyramid  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: <vec4> convolutionPyramidDownscale(<sampler2D> tex, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - CONVOLUTIONPYRAMID_H1: 1.0334, 0.6836, 0.1507

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
 
#ifndef FNC_CONVOLUTIONPYRAMID_DOWNSCALE
#define FNC_CONVOLUTIONPYRAMID_DOWNSCALE
vec4 convolutionPyramidDownscale(sampler2D tex, vec2 st, vec2 pixel) {
    const vec3 h1 = vec3(CONVOLUTIONPYRAMID_H1);

    vec4 color = vec4(0.0);
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel;
            if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0)
                continue;
            color += SAMPLER_FNC(tex, uv) * h1[ absi(dx) ] * h1[ absi(dy) ];
        }
    }

    return color;
}
#endif