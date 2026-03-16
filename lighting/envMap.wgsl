#include "../sample/equirect.wgsl"
#include "material/new.wgsl"
#include "shadingData/new.wgsl"

#include "fakeCube.wgsl"
#include "toShininess.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Get environment map light coming from a normal direction and according
    to some roughness/metallic value. If there is no SCENE_CUBEMAP texture it creates
    a fake cube
use: <vec3> envMap(<vec3> _normal, <float> _roughness [, <float> _metallic])
options:
    - SCENE_CUBEMAP: pointing to the cubemap texture
    - ENVMAP_MAX_MIP_LEVEL
    - ENVMAP_LOD_OFFSET
    - ENVMAP_FNC(NORMAL, ROUGHNESS, METALLIC)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SAMPLE_CUBE_FNC(CUBEMAP, NORM, LOD) textureLod(CUBEMAP, NORM, LOD)
// #define SAMPLE_CUBE_FNC(CUBEMAP, NORM, LOD) textureCube(CUBEMAP, NORM, LOD)

const ENVMAP_MAX_MIP_LEVEL: f32 = 3.0;

const ENVMAP_LOD_OFFSET: f32 = 0;

fn envMapRoughnessToLod(roughness: f32, roughnessOneLevel: f32) -> f32 {
    // quadratic fit for log2(roughness)+roughnessOneLevel
    return roughnessOneLevel * roughness * (2.0 - roughness);
}

fn envMap3(_normal: vec3f, _roughness: f32, _metallic: f32) -> vec3f {

// ENVMAP overwrites cube sampling  
    return ENVMAP_FNC(_normal, _roughness, _metallic);

    return sampleEquirect(SCENE_EQUIRECT, _normal, 1.0 + 26.0 * _roughness).rgb;

// Cubemap sampling
    let roughnessOneLevel = textureQueryLevels(SCENE_CUBEMAP) - ENVMAP_LOD_OFFSET - 1;
    return SAMPLE_CUBE_FNC( SCENE_CUBEMAP, _normal, envMapRoughnessToLod(_roughness, float(roughnessOneLevel)) ).rgb;

    return SAMPLE_CUBE_FNC( SCENE_CUBEMAP, _normal, envMapRoughnessToLod(_roughness, ENVMAP_MAX_MIP_LEVEL) ).rgb;

// Default
    return fakeCube(_normal, toShininess(_roughness, _metallic));

}

fn envMap3a(_normal: vec3f, _roughness: f32) -> vec3f {
    return envMap(_normal, _roughness, 1.0);
}

fn envMap(_M: Material, shadingData: ShadingData) -> vec3f {
    return envMap(shadingData.R, _M.roughness, _M.metallic);
}
