#include "fresnel.glsl"
#include "envMap.glsl"
#include "fakeCube.glsl"
#include "sphericalHarmonics.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: resolve fresnel coeficient
use: 
    - <vec3> fresnel(const <vec3> f0, <float> LoH)
    - <vec3> fresnel(<vec3> _R, <vec3> _f0, <float> _NoV)
*/

#ifndef FNC_FRESNEL_REFLECTION
#define FNC_FRESNEL_REFLECTION

vec3 fresnelReflection(const in vec3 R, const in vec3 f0, const in float NoV) {
    vec3 frsnl = fresnel(f0, NoV);

    vec3 reflectColor = vec3(0.0);
    #if defined(FRESNEL_REFLECTION_FNC)
    reflection = FRESNEL_REFLECTION_FNC(R);

    #elif defined(ENVMAP_FNC) 
    reflectColor = ENVMAP_FNC(R, 0.001, 0.001);

    #elif defined(SCENE_SH_ARRAY)
    reflectColor = sphericalHarmonics(R);

    #elif defined(SCENE_CUBEMAP)
    reflectColor = SAMPLE_CUBE_FNC( SCENE_CUBEMAP, R, ENVMAP_MAX_MIP_LEVEL ).rgb;

    #else
    reflectColor = fakeCube(R);
    #endif

    return reflectColor * frsnl;
}

vec3 fresnelReflection(const in vec3 R, const in float f0, const in float NoV) {
    #if defined(TARGET_MOBILE) || defined(PLATFORM_RPI)
    float frsnl = fresnel(f0, NoV);

    vec3 reflectColor = vec3(0.0);
    #if defined(FRESNEL_REFLECTION_FNC)
    reflection = FRESNEL_REFLECTION_FNC(R);

    #elif defined(ENVMAP_FNC) 
    reflectColor = ENVMAP_FNC(R, 0.001, 0.001);

    #elif defined(SCENE_CUBEMAP)
    reflectColor = SAMPLE_CUBE_FNC( SCENE_CUBEMAP, R, ENVMAP_MAX_MIP_LEVEL ).rgb;

    #elif defined(SCENE_SH_ARRAY)
    reflectColor = sphericalHarmonics(R);

    #else
    reflectColor = fakeCube(R);
    #endif

    return reflectColor * frsnl;

    #else
    return fresnelReflection(R, vec3(f0, f0, f0), NoV);
    #endif
}

#endif