#include "fresnel.hlsl"
#include "fakeCube.hlsl"
#include "sphericalHarmonics.hlsl"
#include "envMap.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: resolve fresnel coeficient
use: 
    - <float3> fresnel(const <float3> f0, <float> LoH)
    - <float3> fresnel(<float3> _R, <float3> _f0, <float> _NoV)
*/

#ifndef FNC_FRESNEL_REFLECTION
#define FNC_FRESNEL_REFLECTION

float3 fresnelReflection(float3 R, float3 f0, float NoV) {
    float3 frsnl = fresnel(f0, NoV);

    float3 reflectColor = float3(0.0, 0.0, 0.0);

    #if defined(FRESNEL_REFLECTION_FNC)
    reflection = FRESNEL_REFLECTION_FNC(R);

    #elif defined(ENVMAP_FNC) 
    reflectColor = ENVMAP_FNC(R, 0.001, 0.001);
    
    #elif defined(SCENE_CUBEMAP)
    reflectColor = SAMPLE_CUBE_FNC( SCENE_CUBEMAP, R, ENVMAP_MAX_MIP_LEVEL).rgb;

    #elif defined(SCENE_SH_ARRAY)
    reflectColor = sphericalHarmonics(R);

    #else
    reflectColor = fakeCube(R);
    #endif

    return reflectColor * frsnl;
}

#endif
