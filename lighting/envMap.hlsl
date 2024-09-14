#include "../sampler.hlsl"
#include "../math/powFast.hlsl"

#include "fakeCube.hlsl"
#include "toShininess.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Get enviroment map light comming from a normal direction and acording
    to some roughness/metallic value. If there is no SCENE_CUBEMAP texture it creates
    a fake cube
use: <float3> envMap(<float3> _normal, <float> _roughness [, <float> _metallic])
options:
    - CUBEMAP: pointing to the cubemap texture
    - ENVMAP_MAX_MIP_LEVEL
    - ENVMAP_LOD_OFFSET
    - ENVMAP_FNC(NORMAL, ROUGHNESS, METALLIC)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SAMPLE_CUBE_FNC
#define SAMPLE_CUBE_FNC(CUBEMAP, NORM, LOD) CUBEMAP.SampleLevel(SAMPLER_TRILINEAR_CLAMP, NORM, LOD)
#endif

#ifndef ENVMAP_LOD_OFFSET
#define ENVMAP_LOD_OFFSET 0
#endif

#ifndef FNC_ENVMAP
#define FNC_ENVMAP

float envMapRoughnessToLod(float roughness, float roughnessOneLevel) {
    // quadratic fit for log2(roughness)+roughnessOneLevel
    return roughnessOneLevel * roughness * (2.0 - roughness);
}

float3 envMap(float3 _normal, float _roughness, float _metallic) {

// ENVMAP overwrites cube sampling  
#if defined(ENVMAP_FNC) 
    return ENVMAP_FNC(_normal, _roughness, _metallic);

// Cubemap sampling
#elif defined(SCENE_CUBEMAP) && !defined(ENVMAP_MAX_MIP_LEVEL)
    uint width, height, levels;
    SCENE_CUBEMAP.GetDimensions(0, width, height, levels);
    float roughnessOneLevel = levels - ENVMAP_LOD_OFFSET - 1;
    return SAMPLE_CUBE_FNC( SCENE_CUBEMAP, _normal, envMapRoughnessToLod(_roughness, roughnessOneLevel)).rgb;
    
#elif defined(SCENE_CUBEMAP)
    return SAMPLE_CUBE_FNC( SCENE_CUBEMAP, _normal, envMapRoughnessToLod(_roughness, ENVMAP_MAX_MIP_LEVEL)).rgb;

// Default
#else
    return fakeCube(_normal, toShininess(_roughness, _metallic));

#endif

}

float3 envMap(const in float3 _normal, const in float _roughness) {
    return envMap(_normal, _roughness, 1.0);
}

float3 envMap(const in Material _M, ShadingData shadingData) {
    return envMap(shadingData.R, _M.roughness, _M.metallic);
}

#endif
