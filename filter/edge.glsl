/*
author: Patricio Gonzalez Vivo
description: Wrapper around three different edge detection algorithms Sobel, Prewitt, and directional Sobel
use: edge(<sampler2D> texture, <vec2> st, <vec2> pixels_scale)
options:
    EDGE_FNC: Edge detection algorithm, defaults to edgeSobel, edgePrewitt & edgeSobel_directional also available
    EDGE_TYPE: Return type, defaults to float
    EDGE_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,POS_UV).r
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef EDGE_FNC
#define EDGE_FNC edgeSobel
#endif

#ifndef EDGE_TYPE
#define EDGE_TYPE float
#endif

#ifndef EDGE_SAMPLER_FNC
#define EDGE_SAMPLER_FNC(POS_UV) texture2D(tex,POS_UV).r
#endif

#include "edge/prewitt.glsl"
#include "edge/sobel.glsl"
#include "edge/sobelDirectional.glsl"

#ifndef FNC_EDGE
#define FNC_EDGE
EDGE_TYPE edge(in sampler2D tex, in vec2 st, in vec2 offset) {
    return EDGE_FNC(tex, st, offset);
}
#endif
