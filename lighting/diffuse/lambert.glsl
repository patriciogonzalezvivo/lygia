/*
author: Patricio Gonzalez Vivo
description: calculate diffuse contribution using lambert equation
use: 
    - <float> diffuseLambert(<vec3> light, <vec3> normal [, <vec3> view, <float> roughness] )
    - <float> diffuseLambert(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_DIFFUSE_LAMBERT
#define FNC_DIFFUSE_LAMBERT
float diffuseLambert(vec3 L, vec3 N) { return max(0.0, dot(N, L)); }
float diffuseLambert(vec3 L, vec3 N, vec3 V, float roughness) { return diffuseLambert(L, N); }
float diffuseLambert(vec3 L, vec3 N, vec3 V, float NoV, float NoL, float roughness) { return diffuseLambert(L, N); }
#endif