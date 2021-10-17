#include "textureShadow.glsl"

/*
author: Patricio Gonzalez Vivo
description: sample shadow map using PCF
use:
    - <float> textureShadowLerp(<sampler2D> depths, <vec2> size, <vec2> uv, <float> compare)
    - <float> textureShadowLerp(<vec3> lightcoord)
options:
    - LIGHT_SHADOWMAP_BIAS
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef LIGHT_SHADOWMAP_BIAS
#define LIGHT_SHADOWMAP_BIAS 0.005
#endif

#ifndef FNC_TEXTURESHADOWLERP
#define FNC_TEXTURESHADOWLERP

float textureShadowLerp(sampler2D depths, vec2 size, vec2 uv, float compare){
    vec2 texelSize = vec2(1.0)/size;
    vec2 f = fract(uv*size+0.5);
    vec2 centroidUV = floor(uv*size+0.5)/size;

    float lb = textureShadow(depths, centroidUV+texelSize*vec2(0.0, 0.0), compare);
    float lt = textureShadow(depths, centroidUV+texelSize*vec2(0.0, 1.0), compare);
    float rb = textureShadow(depths, centroidUV+texelSize*vec2(1.0, 0.0), compare);
    float rt = textureShadow(depths, centroidUV+texelSize*vec2(1.0, 1.0), compare);
    float a = mix(lb, lt, f.y);
    float b = mix(rb, rt, f.y);
    float c = mix(a, b, f.x);
    return c;
}

float textureShadowLerp(vec3 lightcoord) {
#if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE)
    return textureShadowLerp(LIGHT_SHADOWMAP, vec2(LIGHT_SHADOWMAP_SIZE), lightcoord.xy, lightcoord.z - LIGHT_SHADOWMAP_BIAS);
#else
    return 1.0;
#endif
}

#endif