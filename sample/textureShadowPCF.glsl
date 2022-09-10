#include "textureShadowLerp.glsl"

/*
author: Patricio Gonzalez Vivo
description: sample shadow map using PCF
use:
    - <float> textureShadowPCF(<sampler2D> depths, <vec2> size, <vec2> uv, <float> compare)
    - <float> textureShadowPCF(<vec3> lightcoord)
options:
    - LIGHT_SHADOWMAP_BIAS
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef TEXTURESHADOWPCF_SAMPLER_FNC
#define TEXTURESHADOWPCF_SAMPLER_FNC textureShadowLerp
#endif

#ifndef FNC_TEXTURESHADOWPCF
#define FNC_TEXTURESHADOWPCF

float textureShadowPCF(sampler2D depths, vec2 size, vec2 uv, float compare) {
    float result = 0.0;
    for(int x=-2; x<=2; x++){
        for(int y=-2; y<=2; y++){
            vec2 off = vec2(x,y)/size;
            result += TEXTURESHADOWPCF_SAMPLER_FNC(depths, size, uv+off, compare);
        }
    }
    return result/25.0;
}

#endif