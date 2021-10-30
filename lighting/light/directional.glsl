#include "../specular.glsl"
#include "../diffuse.glsl"
#include "falloff.glsl"

/*
author: Patricio Gonzalez Vivo
description: calculate directional light
use: lightDirectional(<vec3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR: in GlslViewer is u_lightColor
    - LIGHT_INTENSITY: in GlslViewer is u_lightIntensity
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef LIGHT_POSITION
#if defined(GLSLVIEWER)
#define LIGHT_POSITION u_light
#else
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif
#endif

#ifndef LIGHT_COLOR
#if defined(GLSLVIEWER)
#define LIGHT_COLOR u_lightColor
#else
#define LIGHT_COLOR vec3(0.5)
#endif
#endif

#ifndef LIGHT_INTENSITY
#if defined(GLSLVIEWER)
#define LIGHT_INTENSITY u_lightIntensity
#else
#define LIGHT_INTENSITY 1.0
#endif
#endif

#ifndef FNC_LIGHT_DIRECTIONAL
#define FNC_LIGHT_DIRECTIONAL

void lightDirectional(vec3 _diffuseColor, vec3 _specularColor, vec3 _N, vec3 _V, float _NoV, float _f0, out vec3 _diffuse, out vec3 _specular) {
    vec3 s = -LIGHT_POSITION;
    float NoL = dot(_N, s);
    float dif = diffuseOrenNayar(s, _N, _V, _NoV, NoL, _roughness);
    float spec = specularCookTorrance(s, _N, _V, _NoV, NoL, _roughness, _comp.f0);
    _diffuse = LIGHT_INTENSITY * (_diffuseColor * LIGHT_COLOR * dif);
    _specular = LIGHT_INTENSITY * (_specularColor * LIGHT_COLOR * spec);
}

#endif