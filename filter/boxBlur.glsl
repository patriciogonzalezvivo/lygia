/*
original_author: Patricio Gonzalez Vivo
description: given a texture return a simple box blured pixel
use: boxBlur(<sampler2D> texture, <vec2> st, <vec2> pixel_offset)
options:
    - BOXBLUR_2D: default to 1D
    - BOXBLUR_ITERATIONS: default 3
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

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
