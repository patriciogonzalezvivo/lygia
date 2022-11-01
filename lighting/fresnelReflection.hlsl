#include "fresnel.hlsl"
#include "envMap.hlsl"
#include "fakeCube.hlsl"
#include "sphericalHarmonics.hlsl"
#include "../color/tonemap.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: resolve fresnel coeficient
use: 
    - <float3> fresnel(const <float3> f0, <float> LoH)
    - <float3> fresnel(<float3> _R, <float3> _f0, <float> _NoV)
*/

#ifndef FNC_FRESNEL_REFLECTION
#define FNC_FRESNEL_REFLECTION

float3 fresnelReflection(float3 _R, float3 _f0, float _NoV) {
    float3 frsnl = fresnel(_f0, _NoV);

    float3 reflectColor = float3(0.0, 0.0, 0.0);
    #if defined(SCENE_SH_ARRAY)
    reflectColor = tonemap( sphericalHarmonics(_R) );
    #else
    reflectColor = fakeCube(_R);
    #endif

    return reflectColor * frsnl;
}

#endif