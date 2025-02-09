#include "fresnel.glsl"
#include "envMap.glsl"
#include "../math/const.glsl"
#include "../lighting/ior.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: "Resolve fresnel coefficient and apply it to a reflection. It can apply\
    \ iridescence to \nusing a formula based on https://www.alanzucconi.com/2017/07/25/the-mathematics-of-thin-film-interference/\n"
use:
    - <vec3> fresnelReflection(<vec3> R, <vec3> f0, <float> NoV)
    - <vec3> fresnelIridescentReflection(<vec3> normal, <vec3> view, <vec3> f0, <vec3> ior1, <vec3> ior2, <float> thickness, <float> roughness)
    - <vec3> fresnelReflection(<Material> _M)
options:
    - FRESNEL_REFLECTION_RGB: <vec3> RGB values of the reflection
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FRESNEL_REFLECTION_RGB
#define FRESNEL_REFLECTION_RGB vec3(612.0, 549.0, 464.0)
#endif

#ifndef FNC_FRESNEL_REFLECTION
#define FNC_FRESNEL_REFLECTION

vec3 fresnelReflection(const in vec3 R, const in vec3 f0, const in float NoV) {
    vec3 frsnl = fresnel(f0, NoV);

    vec3 reflectColor = vec3(0.0);
    #if defined(FRESNEL_REFLECTION_FNC)
    reflectColor = FRESNEL_REFLECTION_FNC(R);
    #else
    reflectColor = envMap(R, 1.0, 0.001);
    #endif

    return reflectColor * frsnl;
}

vec3 fresnelReflection(const in vec3 R, const in vec3 Fr) {
    vec3 reflectColor = vec3(0.0);
    #if defined(FRESNEL_REFLECTION_FNC)
    reflectColor = FRESNEL_REFLECTION_FNC(R);
    #else
    reflectColor = envMap(R, 1.0, 0.001);
    #endif
    return reflectColor * Fr;
}

vec3 fresnelIridescentReflection(vec3 normal, vec3 view, float f0, float ior1, float ior2, float thickness, float roughness, 
                                 inout float Fr) {
    float cos0 = -dot(view, normal);
    Fr = fresnel(f0, cos0);
    float T = 1.0 - Fr;
    float a = ior1/ior2;
    float cosi2 = sqrt(1.0 - a * a * (1.0 - cos0*cos0));
    vec3 shift = 4.0 * PI * (thickness/FRESNEL_REFLECTION_RGB) * ior2 * cosi2 + HALF_PI;
    vec3 irid = Fr * ( 1.0 + T * ( T + 2.0 * cos(shift) ) );
    vec3 ref = envMap(reflect(view, normal), roughness, 0.0);
    return (ref + pow5(ref)) * irid;
}

vec3 fresnelIridescentReflection(vec3 normal, vec3 view, float f0, float ior1, float ior2, float thickness, float roughness) {
    float Fr = 0.0;
    return fresnelIridescentReflection(normal, view, f0, ior1, ior2, thickness, roughness, Fr);
}

vec3 fresnelIridescentReflection(vec3 normal, vec3 view, vec3 f0, vec3 ior1, vec3 ior2, float thickness, float roughness,
                                 inout vec3 Fr) {
    float cos0 = -dot(view, normal);
    Fr = fresnel(f0, cos0);
    vec3 T = 1.0 - Fr;
    vec3 a = ior1/ior2;
    vec3 cosi2 = sqrt(1.0 - a * a * (1.0 - cos0*cos0));
    vec3 shift = 4.0 * PI * (thickness/FRESNEL_REFLECTION_RGB) * ior2 * cosi2 + HALF_PI;
    vec3 irid = Fr * ( 1.0 + T * ( T + 2.0 * cos(shift) ) );
    vec3 ref = envMap(reflect(view, normal), roughness, 0.0);
    return (ref + pow5(ref)) * irid;
}

vec3 fresnelIridescentReflection(vec3 normal, vec3 view, vec3 f0, vec3 ior1, vec3 ior2, float thickness, float roughness) {
    vec3 Fr = vec3(0.0);
    return fresnelIridescentReflection(normal, view, f0, ior1, ior2, thickness, roughness, Fr);
}

vec3 fresnelIridescentReflection(vec3 normal, vec3 view, float ior1, float ior2, float thickness, float roughness) {
    float F0 = (ior2-1.)/(ior2+1.);
    return fresnelIridescentReflection(normal, view, F0 * F0, ior1, ior2, thickness, roughness);
}

vec3 fresnelIridescentReflection(vec3 normal, vec3 view, vec3 ior1, vec3 ior2, float thickness, float roughness) {
    vec3 F0 = (ior2-1.)/(ior2+1.);
    return fresnelIridescentReflection(normal, view, F0 * F0, ior1, ior2, thickness, roughness);
}

vec3 fresnelReflection(Material _M, ShadingData shadingData) {
    vec3 f0 = vec3(0.04, 0.04, 0.04);
    #if defined(SHADING_MODEL_IRIDESCENCE)
    return fresnelIridescentReflection(_M.normal, -shadingData.V, f0, vec3(IOR_AIR, IOR_AIR, IOR_AIR), _M.ior, _M.thickness, _M.roughness);
    #else
    return fresnelReflection(shadingData.R, f0, shadingData.NoV) * (1.0-_M.roughness);
    #endif
}

#endif