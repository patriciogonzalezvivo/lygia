/*
author: Patricio Gonzalez Vivo
description: given a texture return a simple box blured pixel
use: boxBlur(<sampler2D> texture, <vec2> st, <vec2> pixel_offset)
options:
  BOXBLUR_2D: default to 1D
  BOXBLUR_ITERATIONS: default 3
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef BOXBLUR_ITERATIONS
#define BOXBLUR_ITERATIONS 3
#endif

#ifndef BOXBLUR_TYPE
#define BOXBLUR_TYPE vec4
#endif

#ifndef BOXBLUR_SAMPLER_FNC
#define BOXBLUR_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
#endif

#include "boxBlur/1D.glsl"
#include "boxBlur/2D.glsl"
#include "boxBlur/2D_fast9.glsl"

#ifndef FNC_BOXBLUR
#define FNC_BOXBLUR
BOXBLUR_TYPE boxBlur13(in sampler2D tex, in vec2 st, in vec2 offset) {
#ifdef BOXBLUR_2D
  return boxBlur2D(tex, st, offset, 7);
#else
  return boxBlur1D(tex, st, offset, 7);
#endif
}

BOXBLUR_TYPE boxBlur9(in sampler2D tex, in vec2 st, in vec2 offset) {
#ifdef BOXBLUR_2D
  return boxBlur2D_fast9(tex, st, offset);
#else
  return boxBlur1D(tex, st, offset, 5);
#endif
}

BOXBLUR_TYPE boxBlur5(in sampler2D tex, in vec2 st, in vec2 offset) {
#ifdef BOXBLUR_2D
  return boxBlur2D(tex, st, offset, 3);
#else
  return boxBlur1D(tex, st, offset, 3);
#endif
}

vec4 boxBlur(in sampler2D tex, in vec2 st, vec2 offset, const int kernelSize) {
#ifdef BOXBLUR_2D
  return boxBlur2D(tex, st, offset, kernelSize);
#else
  return boxBlur1D(tex, st, offset, kernelSize);
#endif
}

vec4 boxBlur(in sampler2D tex, in vec2 st, vec2 offset) {
  #ifdef BOXBLUR_2D
    return boxBlur2D(tex, st, offset, BOXBLUR_ITERATIONS);
  #else
    return boxBlur1D(tex, st, offset, BOXBLUR_ITERATIONS);
  #endif
}
#endif
