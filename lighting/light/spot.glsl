/*
author: Patricio Gonzalez Vivo
description: calculate spot light
use: lightSpot(<vec3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SURFACE_POSITION: in glslViewer is v_position
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR: in glslViewer is u_lightColor
    - LIGHT_INTENSITY: in glslViewer is  u_lightIntensity

license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#include "../specular.glsl"
#include "../diffuse.glsl"
#include "falloff.glsl"

#ifndef SURFACE_POSITION
#define SURFACE_POSITION v_position
#endif

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

#ifndef FNC_LIGHT_SPOT
#define FNC_LIGHT_SPOT

void lightSpot(vec3 _diffuseColor, vec3 _specularColor, vec3 _N, vec3 _V, float _NoV, float _roughness, float _f0, out vec3 _diffuse, out vec3 _specular) {
    vec3 toLight = LIGHT_POSITION - (SURFACE_POSITION).xyz;
    float toLightLength = length(toLight);
    vec3 s = toLight/toLightLength;

    float angle = acos(dot(-s, light.direction));
    float cutoff1 = radians(clamp(light.spotLightCutoff - max(light.spotLightFactor, 0.01), 0.0, 89.9));
    float cutoff2 = radians(clamp(light.spotLightCutoff, 0.0, 90.0));
    if (angle < cutoff2) {
        float dif = diffuseOrenNayar(s, _N, _V, _NoV, _roughness);
        float fall = falloff(toLightLength, light.spotLightDistance);
        float spec = specularCookTorrance(s, _N, _V, _NoV, _roughness);
        _diffuse = LIGHT_INTENSITY * (_diffuseColor * LIGHT_COLOR * dif * fall) * smoothstep(cutoff2, cutoff1, angle);
        _specular = LIGHT_INTENSITY * (_specularColor * LIGHT_COLOR * spec * fall) * smoothstep(cutoff2, cutoff1, angle);
    }
    else {
        _diffuse = vec3(0.0);
        _specular = vec3(0.0);
    }
}

#endif