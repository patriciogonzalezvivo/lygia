#include "../sampler.hlsl"

/*
contributors:
    - Matt DesLauriers
    - Patricio Gonzalez Vivo
description: Adapted versions from 5, 9 and 13 gaussian fast blur from https://github.com/Jam3/glsl-fast-gaussian-blur
use: gaussianBlur(<SAMPLER_TYPE> texture, <float2> st, <float2> pixel_direction [, const int kernelSize])
options:
    - GAUSSIANBLUR_AMOUNT: gaussianBlur5 gaussianBlur9 gaussianBlur13
    - GAUSSIANBLUR_2D: default to 1D
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef GAUSSIANBLUR_AMOUNT
#define GAUSSIANBLUR_AMOUNT gaussianBlur13
#endif

#ifndef GAUSSIANBLUR_TYPE
#define GAUSSIANBLUR_TYPE float4
#endif

#ifndef GAUSSIANBLUR_SAMPLER_FNC
#define GAUSSIANBLUR_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#include "gaussianBlur/2D.hlsl"
#include "gaussianBlur/1D.hlsl"
#include "gaussianBlur/1D_fast13.hlsl"
#include "gaussianBlur/1D_fast9.hlsl"
#include "gaussianBlur/1D_fast5.hlsl"

#ifndef FNC_GAUSSIANBLUR
#define FNC_GAUSSIANBLUR
GAUSSIANBLUR_TYPE gaussianBlur13(in SAMPLER_TYPE tex, in float2 st, in float2 offset) {
#ifdef GAUSSIANBLUR_2D
    return gaussianBlur2D(tex, st, offset, 7);
#else
    return gaussianBlur1D_fast13(tex, st, offset);
#endif
}

GAUSSIANBLUR_TYPE gaussianBlur9(in SAMPLER_TYPE tex, in float2 st, in float2 offset) {
#ifdef GAUSSIANBLUR_2D
    return gaussianBlur2D(tex, st, offset, 5);
#else
    return gaussianBlur1D_fast9(tex, st, offset);
#endif
}

GAUSSIANBLUR_TYPE gaussianBlur5(in SAMPLER_TYPE tex, in float2 st, in float2 offset) {
#ifdef GAUSSIANBLUR_2D
    return gaussianBlur2D(tex, st, offset, 3);
#else
    return gaussianBlur1D_fast5(tex, st, offset);
#endif
}

GAUSSIANBLUR_TYPE gaussianBlur(in SAMPLER_TYPE tex, in float2 st, in float2 offset, const int kernelSize) {
#ifdef GAUSSIANBLUR_2D
    return gaussianBlur2D(tex, st, offset, kernelSize);
#else
    return gaussianBlur1D(tex, st, offset, kernelSize);
#endif
}

GAUSSIANBLUR_TYPE gaussianBlur(in SAMPLER_TYPE tex, in float2 st, in float2 offset) {
    return GAUSSIANBLUR_AMOUNT(tex, st, offset);
}
#endif
