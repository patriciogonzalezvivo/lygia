/*
author: Matt DesLauriers
description: Performs a smoothstep using standard derivatives for anti-aliased edges at any level of magnification. From https://github.com/glslify/glsl-aastep
use: aastep(<float> threshold, <float> value)
option:
    AA_EDGE: in the absence of derivatives you can specify the antialiasing factor
license: |
    The MIT License (MIT) Copyright (c) 2015 stackgl
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_AASTEP
#define FNC_AASTEP
#ifdef GL_OES_standard_derivatives
#extension GL_OES_standard_derivatives : enable
#endif

#ifndef AA_EDGE
#define AA_EDGE .007
#endif

float aastep(float threshold, float value) {
    #ifdef GL_OES_standard_derivatives
    float afwidth = 0.7 * length(vec2(dFdx(value), dFdy(value)));
    #else
    float afwidth = AA_EDGE;
    #endif
    return smoothstep(threshold-afwidth, threshold+afwidth, value);
}
#endif