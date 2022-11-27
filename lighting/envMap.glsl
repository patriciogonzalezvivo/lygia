
#include "../math/powFast.glsl"
#include "../color/tonemap.glsl"

#include "fakeCube.glsl"
#include "toShininess.glsl"
#include "sphericalHarmonics.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: get enviroment map light comming from a normal direction and acording to some roughness/metallic value. If there is no SCENE_CUBEMAP texture it creates a fake cube
use: <vec3> envMap(<vec3> _normal, <float> _roughness [, <float> _metallic])
options:
    - SCENE_CUBEMAP: pointing to the cubemap texture
    - ENVMAP_MAX_MIP_LEVEL: defualt 8
    - ENVMAP_FNC(NORMAL, ROUGHNESS, METALLIC)
*/

#ifndef SAMPLE_CUBE_FNC
#define SAMPLE_CUBE_FNC(CUBEMAP, NORM, LOD) textureCube(CUBEMAP, NORM, LOD)
#endif

#ifndef ENVMAP_MAX_MIP_LEVEL
#define ENVMAP_MAX_MIP_LEVEL 3.0
#endif

#ifndef FNC_ENVMAP
#define FNC_ENVMAP
vec3 envMap(const in vec3 _normal, const in float _roughness, const in float _metallic) {

// ENVMAP overwrites cube sampling  
#if defined(ENVMAP_FNC) 
    return ENVMAP_FNC(_normal, _roughness, _metallic);

// Cubemap sampling - spherical harmonics
#elif defined(SCENE_CUBEMAP) && defined(SCENE_SH_ARRAY) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
    return mix(
        SAMPLE_CUBE_FNC( SCENE_CUBEMAP, _normal, (ENVMAP_MAX_MIP_LEVEL * _roughness) ).rgb,
        sphericalHarmonics(_normal),
        _roughness * _roughness * _roughness
    );

// Cubemap sampling
#elif defined(SCENE_CUBEMAP)
    return SAMPLE_CUBE_FNC( SCENE_CUBEMAP, _normal, (ENVMAP_MAX_MIP_LEVEL * _roughness) ).rgb;

// Default
#else
    return fakeCube(_normal, toShininess(_roughness, _metallic));

#endif
}

vec3 envMap(const in vec3 _normal, const in float _roughness) {
    return envMap(_normal, _roughness, 1.0);
}
#endif