#include "fresnel.glsl"
#include "envMap.glsl"
#include "../color/tonemap.glsl"
#include "sphericalHarmonics.glsl"

/*
contributors: Patricio Gonzalez Vivo
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

    // #elif defined(SCENE_SH_ARRAY)
    // reflectColor = tonemap( sphericalHarmonics(R) );

    #else
    reflectColor = envMap(R, 1.0, 0.001);

    #endif

    return reflectColor * frsnl;
}

vec3 fresnelIridescentReflection(vec3 R, vec3 f0, float NoV, float thickness, float ior0, float ior1, float ior2, float roughness) {
    vec3 frsnl = fresnelIridescent(f0, NoV, thickness, ior0, ior1, ior2, roughness);

    vec3 reflectColor = vec3(0.0);

    #if defined(FRESNEL_REFLECTION_FNC)
    reflection = FRESNEL_REFLECTION_FNC(R);

    // #elif defined(SCENE_SH_ARRAY)
    // reflectColor = tonemap( sphericalHarmonics(R) );

    #else
    reflectColor = envMap(R, roughness, 0.00001);

    #endif

    return reflectColor * frsnl;
}


#ifdef STR_MATERIAL
vec3 fresnelReflection(const in Material _M) {
    return fresnelReflection(_M.R, _M.f0, _M.NoV);
}
#endif

#endif