#include "fresnel.hlsl"
#include "envMap.hlsl"
#include "../math/const.hlsl"
#include "../lighting/ior.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: "Resolve fresnel coeficient and apply it to a reflection. It can apply\
    \ iridescence to \nusing a formula based on https://www.alanzucconi.com/2017/07/25/the-mathematics-of-thin-film-interference/\n"
use:
    - <float3> fresnelReflection(<float3> R, <float3> f0, <float> NoV)
    - <float3> fresnelIridescentReflection(<float3> normal, <float3> view, <float3> f0, <float3> ior1, <float3> ior2, <float> thickness, <float> roughness)
    - <float3> fresnelReflection(<Material> _M)
options:
    - FRESNEL_REFLECTION_RGB: <float3> RGB values of the reflection
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FRESNEL_REFLECTION_RGB
#define FRESNEL_REFLECTION_RGB float3(612.0, 549.0, 464.0)
#endif

#ifndef FNC_FRESNEL_REFLECTION
#define FNC_FRESNEL_REFLECTION

float3 fresnelReflection(const in float3 R, const in float3 f0, const in float NoV)
{
    float3 frsnl = fresnel(f0, NoV);

    float3 reflectColor = float3(0.0, 0.0, 0.0);
#if defined(FRESNEL_REFLECTION_FNC)
    reflectColor = FRESNEL_REFLECTION_FNC(R);
#else
    reflectColor = envMap(R, 1.0, 0.001);
#endif

    return reflectColor * frsnl;
}

float3 fresnelReflection(const in float3 R, const in float3 Fr)
{
    float3 reflectColor = float3(0.0, 0.0, 0.0);
#if defined(FRESNEL_REFLECTION_FNC)
    reflectColor = FRESNEL_REFLECTION_FNC(R);
#else
    reflectColor = envMap(R, 1.0, 0.001);
#endif
    return reflectColor * Fr;
}

float3 fresnelIridescentReflection(float3 normal, float3 view, float f0, float ior1, float ior2, float thickness, float roughness,
                                 inout float Fr)
{
    float cos0 = -dot(view, normal);
    Fr = fresnel(f0, cos0);
    float T = 1.0 - Fr;
    float a = ior1 / ior2;
    float cosi2 = sqrt(1.0 - a * a * (1.0 - cos0 * cos0));
    float3 shift = 4.0 * PI * (thickness / FRESNEL_REFLECTION_RGB) * ior2 * cosi2 + HALF_PI;
    float3 irid = Fr * (1.0 + T * (T + 2.0 * cos(shift)));
    float3 ref = envMap(reflect(view, normal), roughness, 0.0);
    return (ref + pow5(ref)) * irid;
}

float3 fresnelIridescentReflection(float3 normal, float3 view, float f0, float ior1, float ior2, float thickness, float roughness)
{
    float Fr = 0.0;
    return fresnelIridescentReflection(normal, view, f0, ior1, ior2, thickness, roughness, Fr);
}

float3 fresnelIridescentReflection(float3 normal, float3 view, float3 f0, float3 ior1, float3 ior2, float thickness, float roughness,
                                 inout float3 Fr)
{
    float cos0 = -dot(view, normal);
    Fr = fresnel(f0, cos0);
    float3 T = 1.0 - Fr;
    float3 a = ior1 / ior2;
    float3 cosi2 = sqrt(1.0 - a * a * (1.0 - cos0 * cos0));
    float3 shift = 4.0 * PI * (thickness / FRESNEL_REFLECTION_RGB) * ior2 * cosi2 + HALF_PI;
    float3 irid = Fr * (1.0 + T * (T + 2.0 * cos(shift)));
    float3 ref = envMap(reflect(view, normal), roughness, 0.0);
    return (ref + pow5(ref)) * irid;
}

float3 fresnelIridescentReflection(float3 normal, float3 view, float3 f0, float3 ior1, float3 ior2, float thickness, float roughness)
{
    float3 Fr = float3(0.0, 0.0, 0.0);
    return fresnelIridescentReflection(normal, view, f0, ior1, ior2, thickness, roughness, Fr);
}

float3 fresnelIridescentReflection(float3 normal, float3 view, float ior1, float ior2, float thickness, float roughness)
{
    float F0 = (ior2 - 1.) / (ior2 + 1.);
    return fresnelIridescentReflection(normal, view, F0 * F0, ior1, ior2, thickness, roughness);
}

float3 fresnelIridescentReflection(float3 normal, float3 view, float3 ior1, float3 ior2, float thickness, float roughness)
{
    float3 F0 = (ior2 - 1.) / (ior2 + 1.);
    return fresnelIridescentReflection(normal, view, F0 * F0, ior1, ior2, thickness, roughness);
}

float3 fresnelReflection(Material _M, ShadingData shadingData) {
    float3 f0 = float3(0.04, 0.04, 0.04);
    #if defined(SHADING_MODEL_IRIDESCENCE)
    return fresnelIridescentReflection(_M.normal, -shadingData.V, f0, float3(IOR_AIR, IOR_AIR, IOR_AIR), _M.ior, _M.thickness, _M.roughness);
    #else
    return fresnelReflection(shadingData.R, f0, shadingData.NoV) * (1.0-_M.roughness);
    #endif
}

#endif