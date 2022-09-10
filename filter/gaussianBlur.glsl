/*
author: [Matt DesLauriers, Patricio Gonzalez Vivo]
description: adapted versions from 5, 9 and 13 gaussian fast blur from https://github.com/Jam3/glsl-fast-gaussian-blur
use: gaussianBlur(<sampler2D> texture, <vec2> st, <vec2> pixel_direction [, const int kernelSize])
options:
    - GAUSSIANBLUR_AMOUNT: gaussianBlur5 gaussianBlur9 gaussianBlur13 
    - GAUSSIANBLUR_2D: default to 1D
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef GAUSSIANBLUR_AMOUNT
#define GAUSSIANBLUR_AMOUNT gaussianBlur13
#endif

#ifndef GAUSSIANBLUR_TYPE
#define GAUSSIANBLUR_TYPE vec4
#endif

#ifndef GAUSSIANBLUR_SAMPLER_FNC
#define GAUSSIANBLUR_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif

#include "gaussianBlur/2D.glsl"
#include "gaussianBlur/1D.glsl"
#include "gaussianBlur/1D_fast13.glsl"
#include "gaussianBlur/1D_fast9.glsl"
#include "gaussianBlur/1D_fast5.glsl"

#ifndef FNC_GAUSSIANBLUR
#define FNC_GAUSSIANBLUR
GAUSSIANBLUR_TYPE gaussianBlur13(in sampler2D tex, in vec2 st, in vec2 offset) {
#ifdef GAUSSIANBLUR_2D
    return gaussianBlur2D(tex, st, offset, 7);
#else
    return gaussianBlur1D_fast13(tex, st, offset);
#endif
}

GAUSSIANBLUR_TYPE gaussianBlur9(in sampler2D tex, in vec2 st, in vec2 offset) {
#ifdef GAUSSIANBLUR_2D
    return gaussianBlur2D(tex, st, offset, 5);
#else
    return gaussianBlur1D_fast9(tex, st, offset);
#endif
}

GAUSSIANBLUR_TYPE gaussianBlur5(in sampler2D tex, in vec2 st, in vec2 offset) {
#ifdef GAUSSIANBLUR_2D
    return gaussianBlur2D(tex, st, offset, 3);
#else
    return gaussianBlur1D_fast5(tex, st, offset);
#endif
}

GAUSSIANBLUR_TYPE gaussianBlur(in sampler2D tex, in vec2 st, in vec2 offset, const int kernelSize) {
#ifdef GAUSSIANBLUR_2D
    return gaussianBlur2D(tex, st, offset, kernelSize);
#else
    return gaussianBlur1D(tex, st, offset, kernelSize);
#endif
}

GAUSSIANBLUR_TYPE gaussianBlur(in sampler2D tex, in vec2 st, in vec2 offset) {
    return GAUSSIANBLUR_AMOUNT(tex, st, offset);
}
#endif
