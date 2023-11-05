#include "../sample.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: sampler shadowMap 
use: 
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <vec4|vec3> _coord)
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <vec2> _coord , float compare)
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <vec2> _shadowMapSize, <vec2> _coord , float compare)
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef SAMPLERSHADOW_FNC
#define SAMPLERSHADOW_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r
#endif

#ifndef FNC_SAMPLESHADOW
#define FNC_SAMPLESHADOW

float sampleShadow(in SAMPLER_TYPE shadowMap, in vec4 _coord) {
    vec3 shadowCoord = _coord.xyz / _coord.w;
    return SAMPLERSHADOW_FNC(shadowMap, shadowCoord.xy);
}

float sampleShadow(in SAMPLER_TYPE shadowMap, in vec3 _coord) {
    return sampleShadow(shadowMap, vec4(_coord, 1.0));
}

float sampleShadow(in SAMPLER_TYPE shadowMap, in vec2 uv, in float compare) {
    return step(compare, SAMPLERSHADOW_FNC(shadowMap, uv) );
}

float sampleShadow(in SAMPLER_TYPE shadowMap, in vec2 size, in vec2 uv, in float compare) {
    return sampleShadow(shadowMap, uv, compare);
}

#endif