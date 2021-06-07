/*
author: Patricio Gonzalez Vivo
description: simple laplacian filter
use: laplace(<sampler2D> texture, <vec2> st, <vec2> pixels_scale)
options:
    LAPLACE_TYPE: Return type, defaults to float
    LAPLACE_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,POS_UV).r
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef LAPLACE_TYPE
#define LAPLACE_TYPE vec3
#endif

#ifndef LAPLACE_SAMPLER_FNC
#define LAPLACE_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV).rgb
#endif

#ifndef FNC_LAPLACE
#define FNC_LAPLACE
LAPLACE_TYPE laplace(sampler2D tex, vec2 st, vec2 pixel) {
    return    LAPLACE_SAMPLER_FNC(st) * 4.0
            - LAPLACE_SAMPLER_FNC(st + vec2(-1.0,  0.0) * pixel)
            - LAPLACE_SAMPLER_FNC(st + vec2( 0.0, -1.0) * pixel)
            - LAPLACE_SAMPLER_FNC(st + vec2( 0.0,  1.0) * pixel)
            - LAPLACE_SAMPLER_FNC(st + vec2( 1.0,  0.0) * pixel);
}
#endif