/*
author: Patricio Gonzalez Vivo
description: calculate diffuse contribution using Oren and Nayar equation https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
use: 
    - <float> diffuseOrenNayar(<vec3> light, <vec3> normal, <vec3> view, <float> roughness )
    - <float> diffuseOrenNayar(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_DIFFUSE_ORENNAYAR
#define FNC_DIFFUSE_ORENNAYAR

float diffuseOrenNayar(vec3 L, vec3 N, vec3 V, float NoV, float NoL, float roughness) {
    float LoV = dot(L, V);
    
    float s = LoV - NoL * NoV;
    float t = mix(1.0, max(NoL, NoV), step(0.0, s));

    float sigma2 = roughness * roughness;
    float A = 1.0 + sigma2 * (1.0 / (sigma2 + 0.13) + 0.5 / (sigma2 + 0.33));
    float B = 0.45 * sigma2 / (sigma2 + 0.09);

    return max(0.0, NoL) * (A + B * s / t);
}

float diffuseOrenNayar(vec3 L, vec3 N, vec3 V, float roughness) {
    float NoV = max(dot(N, V), 0.001);
    float NoL = max(dot(N, L), 0.001);
    return diffuseOrenNayar(L, N, V, NoV, NoL, roughness);
}

#endif