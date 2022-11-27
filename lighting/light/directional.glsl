#include "../specular.glsl"
#include "../diffuse.glsl"
#include "falloff.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: calculate directional light
use: lightDirectional(<vec3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_DIRECTION
    - LIGHT_COLOR: in GlslViewer is u_lightColor
    - LIGHT_INTENSITY: in GlslViewer is u_lightIntensity
*/

#ifndef LIGHT_POSITION
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR vec3(0.5)
#endif

#ifndef LIGHT_INTENSITY
#define LIGHT_INTENSITY 1.0
#endif

#ifndef FNC_LIGHT_DIRECTIONAL
#define FNC_LIGHT_DIRECTIONAL
void lightDirectional(const in vec3 _diffuseColor, const in vec3 _specularColor, const in vec3 _N, const in vec3 _V, const in float _NoV, const in float _roughness, const in float _f0, const in float _shadow, inout vec3 _diffuse, inout vec3 _specular) {
    #ifdef LIGHT_DIRECTION
    vec3    D = normalize(LIGHT_DIRECTION);
    #else 
    vec3    D = normalize(LIGHT_POSITION);
    #endif
    float NoL = dot(_N, D);
    float dif = diffuse(D, _N, _V, _NoV, NoL, _roughness);
    float spec = specular(D, _N, _V, _NoV, NoL, _roughness, _f0);
    _diffuse  += max(vec3(0.0), LIGHT_INTENSITY * (_diffuseColor * LIGHT_COLOR * dif) * _shadow);
    _specular += max(vec3(0.0), LIGHT_INTENSITY * (_specularColor * LIGHT_COLOR * spec) * _shadow);
}
#endif