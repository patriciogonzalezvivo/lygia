/*
original_author: Patricio Gonzalez Vivo
description: calculate point light
use: lightPoint(<vec3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SURFACE_POSITION: in glslViewer is v_position
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR: in glslViewer is u_lightColor
    - LIGHT_INTENSITY: in glslViewer is  u_lightIntensity
    - LIGHT_FALLOFF: in glslViewer is u_lightFalloff
*/

#include "../specular.glsl"
#include "../diffuse.glsl"
#include "falloff.glsl"

#ifndef SURFACE_POSITION
#define SURFACE_POSITION vec3(0.0, 0.0, 0.0)
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION  vec3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR    vec3(0.5, 0.5, 0.5)
#endif

#ifndef LIGHT_INTENSITY
#define LIGHT_INTENSITY 1.0
#endif

#ifndef LIGHT_FALLOFF
#define LIGHT_FALLOFF   0.0
#endif

#ifndef FNC_LIGHT_POINT
#define FNC_LIGHT_POINT

void lightPoint(const in vec3 _diffuseColor, const in vec3 _specularColor, const in vec3 _N, const in vec3 _V, const in float _NoV, const in float _roughness, const in float _f0, const in float _shadow, inout vec3 _diffuse, inout vec3 _specular) {
    vec3 toLight = LIGHT_POSITION - (SURFACE_POSITION).xyz;
    float toLightLength = length(toLight);
    vec3 s = toLight/toLightLength;

    float NoL = dot(_N, s);

    float dif = diffuse(s, _N, _V, _NoV, NoL, _roughness);// * ONE_OVER_PI;
    float spec = specular(s, _N, _V, _NoV, NoL, _roughness, _f0);

    vec3 lightContribution = LIGHT_COLOR * LIGHT_INTENSITY * _shadow;
    #ifdef LIGHT_FALLOFF
    if (LIGHT_FALLOFF > 0.0)
        lightContribution *= falloff(toLightLength, LIGHT_FALLOFF);
    #endif

    _diffuse +=  max(vec3(0.0), _diffuseColor * lightContribution * dif);
    _specular += max(vec3(0.0), _specularColor * lightContribution * spec);
}

void lightPoint(const in vec3 _diffuseColor, const in vec3 _specularColor, const in vec3 _N, const in vec3 _V, const in float _NoV, const in float _roughness, const in float _f0, inout vec3 _diffuse, inout vec3 _specular) {
    lightPoint(_diffuseColor, _specularColor, _N, _V,  _NoV, _roughness, _f0, 1.0, _diffuse, _specular);
}

#endif
