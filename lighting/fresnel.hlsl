#include "common/schlick.hlsl"

#include "fakeCube.hlsl"
#include "sphericalHarmonics.hlsl"
#include "../color/tonemap.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: resolve fresnel coeficient
use: 
    - <float3> fresnel(const <float3> f0, <float> LoH)
    - <float3> fresnel(<float3> _R, <float3> _f0, <float> _NoV)
*/

#ifndef FNC_FRESNEL
#define FNC_FRESNEL

float3 fresnel(const float3 f0, float LoH) {
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI)
    return schlick(f0, 1.0, LoH);
#else
    float f90 = saturate(dot(f0, float3(50.0, 50.0, 50.0) * 0.33));
    return schlick(f0, f90, LoH);
#endif
}

// float fresnelf(float3 V, float3 N, float R0) {
//     float cosAngle = 1.0-max(dot(V, N), 0.0);
//     float result = cosAngle * cosAngle;
//     result = result * result;
//     result = result * cosAngle;
//     result = clamp(result * (1.0 - R0) + R0, 0.0, 1.0);
//     return result;
// }

#endif