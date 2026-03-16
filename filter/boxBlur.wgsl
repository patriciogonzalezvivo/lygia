#include "../sampler.wgsl"

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

const BOXBLUR_ITERATIONS: f32 = 3;

// #define BOXBLUR_TYPE vec4

// #define BOXBLUR_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

#include "boxBlur/1D.wgsl"
#include "boxBlur/2D.wgsl"
#include "boxBlur/2D_fast9.wgsl"

BOXBLUR_TYPE boxBlur13(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
  return boxBlur2D(tex, st, offset, 7);
  return boxBlur1D(tex, st, offset, 7);
}

BOXBLUR_TYPE boxBlur9(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
  return boxBlur2D_fast9(tex, st, offset);
  return boxBlur1D(tex, st, offset, 5);
}

BOXBLUR_TYPE boxBlur5(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
  return boxBlur2D(tex, st, offset, 3);
  return boxBlur1D(tex, st, offset, 3);
}

BOXBLUR_TYPE boxBlur(in SAMPLER_TYPE tex, in vec2 st, vec2 offset, const int kernelSize) {
  return boxBlur2D(tex, st, offset, kernelSize);
  return boxBlur1D(tex, st, offset, kernelSize);
}

BOXBLUR_TYPE boxBlur(in SAMPLER_TYPE tex, in vec2 st, vec2 offset) {
    return boxBlur2D(tex, st, offset, BOXBLUR_ITERATIONS);
    return boxBlur1D(tex, st, offset, BOXBLUR_ITERATIONS);
}
