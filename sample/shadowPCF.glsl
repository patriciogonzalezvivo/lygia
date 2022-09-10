#include "shadowLerp.glsl"

/*
author: Patricio Gonzalez Vivo
description: sample shadow map using PCF
use:
    - <float> sampleShadowPCF(<sampler2D> depths, <vec2> size, <vec2> uv, <float> compare)
    - <float> sampleShadowPCF(<vec3> lightcoord)
options:
    - LIGHT_SHADOWMAP_BIAS
    - SAMPLESHADOWPCF_SAMPLER_FNC
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef SAMPLESHADOWPCF_SAMPLER_FNC
#define SAMPLESHADOWPCF_SAMPLER_FNC sampleShadowLerp
#endif

#ifndef FNC_SAMPLESHADOWPCF
#define FNC_SAMPLESHADOWPCF

float sampleShadowPCF(sampler2D depths, vec2 size, vec2 uv, float compare) {
    vec2 pixel = 1.0/size;
    float result = 0.0;
    for (float x= -2.0; x <= 2.0; x++)
        for (float y= -2.0; y <= 2.0; y++) 
            result += SAMPLESHADOWPCF_SAMPLER_FNC(depths, size, uv + vec2(x,y) * pixel, compare);
    return result/25.0;
}

#endif