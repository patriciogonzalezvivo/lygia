#include "../sampler.glsl"

/*
contributors: [Morgan McGuire, Kyle Whitson]
description: 3x3 and 5x5 median filter, adapted from "A Fast, Small-Radius GPU Median Filter" by Morgan McGuire in ShaderX6 https://casual-effects.com/research/McGuire2008Median/index.html
use: median(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel)
options:
    - MEDIAN_AMOUNT: median3 (3x3) median5 (5x5)
    - MEDIAN_TYPE: default vec4
    - MEDIAN_SAMPLER_FNC(TEX, UV): default texture2D(TEX, UV)
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license: >
    Copyright (c) Morgan McGuire and Williams College, 2006. All rights reserved.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
examples:
    - /shaders/filter_median2D.frag
*/

#ifndef MEDIAN_AMOUNT
#define MEDIAN_AMOUNT median5
#endif

#ifndef MEDIAN_TYPE
#define MEDIAN_TYPE vec4
#endif

#ifndef MEDIAN_SAMPLER_FNC
#define MEDIAN_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#include "median/2D_fast3.glsl"
#include "median/2D_fast5.glsl"

#ifndef FNC_MEDIAN
#define FNC_MEDIAN
MEDIAN_TYPE median3(in SAMPLER_TYPE tex, in vec2 st, in vec2 radius) {
    return median2D_fast3(tex, st, radius);
}

MEDIAN_TYPE median5(in SAMPLER_TYPE tex, in vec2 st, in vec2 radius) {
    return median2D_fast5(tex, st, radius);
}

MEDIAN_TYPE median(in SAMPLER_TYPE tex, in vec2 st, in vec2 radius) {
    return MEDIAN_AMOUNT(tex, st, radius);
}
#endif
