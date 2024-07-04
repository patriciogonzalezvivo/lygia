#include "common/schlick.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Resolve fresnel coeficient
use:
    - <float3> fresnel(const <float3> f0, <float> NoV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FRESNEL
#define FNC_FRESNEL

float3 fresnel(const in float3 f0, float3 normal, float3 view)
{
    return schlick(f0, 1.0, dot(view, normal));
}

float3 fresnel(const float3 f0, float NoV)
{
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI)
    return schlick(f0, 1.0, NoV);
#else
    float f90 = saturate(dot(f0, float3(50.0, 50.0, 50.0) * 0.33));
    return schlick(f0, f90, NoV);
#endif
}

float fresnel(const in float f0, const in float NoV)
{
    return schlick(f0, 1.0, NoV);
}


#endif