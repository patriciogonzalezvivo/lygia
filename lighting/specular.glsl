#include "specular/phong.glsl"
#include "specular/blinnPhong.glsl"
#include "specular/cookTorrance.glsl"
#include "specular/gaussian.glsl"
#include "specular/beckmann.glsl"
#include "specular/ggx.glsl"

/*
author: Patricio Gonzalez Vivo
description: calculate specular contribution
use: 
    - specular(<vec3> L, <vec3> N, <vec3> V, <float> roughne#ifndef TONEMAP_FNC
#define TONEMAP_FNC tonemapReinhard
#endifss [, <float> fresnel])
    - specular(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness, <float> fresnel)
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)

license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef SPECULAR_FNC
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
#define SPECULAR_FNC specularBlinnPhongRoughnes
#else
#define SPECULAR_FNC specularCookTorrance
#endif  
#endif

#ifndef FNC_SPECULAR
#define FNC_SPECULAR
float specular(vec3 L, vec3 N, vec3 V, float roughness) { return SPECULAR_FNC(L, N, V, roughness); }
float specular(vec3 L, vec3 N, vec3 V, float roughness, float fresnel) { return SPECULAR_FNC(L, N, V, roughness, fresnel); }
float specular(vec3 L, vec3 N, vec3 V, float NoV, float NoL, float roughness, float fresnel) { return SPECULAR_FNC(L, N, V, NoV, NoL, roughness, fresnel); }
#endif