/*
original_author: [Matt DesLauriers, Patricio Gonzalez Vivo]
description: adapted versions from 5, 9 and 13 gaussian fast blur from https://github.com/Jam3/glsl-fast-gaussian-blur
use: gaussianBlur(<sampler2D> texture, <vec2> st, <vec2> pixel_direction [, const int kernelSize])
options:
    - GAUSSIANBLUR_AMOUNT: gaussianBlur5 gaussianBlur9 gaussianBlur13 
    - GAUSSIANBLUR_2D: default to 1D
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
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
