#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Given a texture it performs a moving average or box blur. 
    Which simply averages the pixel values in a KxK window. 
    This is a very common image processing technique that can be used to smooth out noise.
use: boxBlur(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel_offset)
options:
    - BOXBLUR_2D: default to 1D
    - BOXBLUR_ITERATIONS: default 3
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
examples:
    - /shaders/filter_boxBlur2D.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef BOXBLUR_ITERATIONS
#define BOXBLUR_ITERATIONS 3
#endif

#ifndef BOXBLUR_TYPE
#define BOXBLUR_TYPE vec4
#endif

#ifndef BOXBLUR_SAMPLER_FNC
#define BOXBLUR_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#include "boxBlur/1D.glsl"
#include "boxBlur/2D.glsl"
#include "boxBlur/2D_fast9.glsl"

#ifndef FNC_BOXBLUR
#define FNC_BOXBLUR
BOXBLUR_TYPE boxBlur13(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
#ifdef BOXBLUR_2D
  return boxBlur2D(tex, st, offset, 7);
#else
  return boxBlur1D(tex, st, offset, 7);
#endif
}

BOXBLUR_TYPE boxBlur9(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
#ifdef BOXBLUR_2D
  return boxBlur2D_fast9(tex, st, offset);
#else
  return boxBlur1D(tex, st, offset, 5);
#endif
}

BOXBLUR_TYPE boxBlur5(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
#ifdef BOXBLUR_2D
  return boxBlur2D(tex, st, offset, 3);
#else
  return boxBlur1D(tex, st, offset, 3);
#endif
}

BOXBLUR_TYPE boxBlur(in SAMPLER_TYPE tex, in vec2 st, vec2 offset, const int kernelSize) {
#ifdef BOXBLUR_2D
  return boxBlur2D(tex, st, offset, kernelSize);
#else
  return boxBlur1D(tex, st, offset, kernelSize);
#endif
}

BOXBLUR_TYPE boxBlur(in SAMPLER_TYPE tex, in vec2 st, vec2 offset) {
  #ifdef BOXBLUR_2D
    return boxBlur2D(tex, st, offset, BOXBLUR_ITERATIONS);
  #else
    return boxBlur1D(tex, st, offset, BOXBLUR_ITERATIONS);
  #endif
}
#endif
