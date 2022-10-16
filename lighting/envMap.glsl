
#include "../math/powFast.glsl"
#include "../color/tonemap.glsl"

#include "fakeCube.glsl"
#include "toShininess.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: get enviroment map light comming from a normal direction and acording to some roughness/metallic value. If there is no SCENE_CUBEMAP texture it creates a fake cube
use: <vec3> envMap(<vec3> _normal, <float> _roughness [, <float> _metallic])
options:
    - SCENE_CUBEMAP: pointing to the cubemap texture
    - ENVMAP_MAX_MIP_LEVEL: defualt 8
*/

#ifndef SAMPLE_CUBE_FNC
#define SAMPLE_CUBE_FNC(CUBEMAP, NORM, LOD) textureCube(CUBEMAP, NORM, LOD)
#endif

#ifndef ENVMAP_MAX_MIP_LEVEL
#define ENVMAP_MAX_MIP_LEVEL 3.0
#endif

#ifndef ENVMAP_FNC
#if defined(SCENE_CUBEMAP)
#define ENVMAP_FNC(NORMAL, ROUGHNESS) SAMPLE_CUBE_FNC( SCENE_CUBEMAP, _normal, ENVMAP_MAX_MIP_LEVEL * _roughness).rgb;
#else
#define ENVMAP_FNC(NORMAL, ROUGHNESS) fakeCube(NORMAL, toShininess(_roughness, ROUGHNESS));
// #define ENVMAP_FNC(NORMAL, ROUGHNESS) fakeCube(NORMAL, ROUGHNESS);
#endif
#endif

#ifndef FNC_ENVMAP
#define FNC_ENVMAP
vec3 envMap(vec3 _normal, float _roughness, float _metallic) {
    return ENVMAP_FNC(_normal, _roughness);
// #if defined(SCENE_CUBEMAP)
//     float lod = ENVMAP_MAX_MIP_LEVEL * _roughness;
//     return SAMPLE_CUBE_FNC( SCENE_CUBEMAP, _normal, lod).rgb;
// #else
    // return fakeCube(_normal, toShininess(_roughness, _metallic));
// #endif
}

vec3 envMap(vec3 _normal, float _roughness) {
    return envMap(_normal, _roughness, 1.0);
}
#endif