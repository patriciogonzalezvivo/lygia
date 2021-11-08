#include "common/schlick.glsl"
#include "envMap.glsl"
#include "fakeCube.glsl"
#include "sphericalHarmonics.glsl"
#include "../color/tonemap.glsl"
#include "../math/saturate.glsl"

/*
author: Patricio Gonzalez Vivo
description: resolve fresnel coeficient
use: 
    - <vec3> fresnel(const <vec3> f0, <float> LoH)
    - <vec3> fresnel(<vec3> _R, <vec3> _f0, <float> _NoV)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_FRESNEL
#define FNC_FRESNEL

vec3 fresnel(const vec3 f0, float LoH) {
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
    return schlick(f0, 1.0, LoH);
#else
    float f90 = saturate(dot(f0, vec3(50.0 * 0.33)));
    return schlick(f0, f90, LoH);
#endif
}

vec3 fresnel(vec3 _R, vec3 _f0, float _NoV) {
    vec3 frsnl = fresnel(_f0, _NoV);

    vec3 reflectColor = vec3(0.0);
    #if defined(SCENE_SH_ARRAY)
    reflectColor = tonemap( sphericalHarmonics(_R) );
    #else
    reflectColor = fakeCube(_R);
    #endif

    return reflectColor * frsnl;
}

float fresnelf(vec3 V, vec3 N, float R0) {
    float cosAngle = 1.0-max(dot(V, N), 0.0);
    float result = cosAngle * cosAngle;
    result = result * result;
    result = result * cosAngle;
    result = clamp(result * (1.0 - R0) + R0, 0.0, 1.0);
    return result;
}

#endif