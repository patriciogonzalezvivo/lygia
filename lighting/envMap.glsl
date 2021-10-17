
#include "../math/powFast.glsl"
#include "../color/tonemap.glsl"

#include "fakeCube.glsl"
#include "toShininess.glsl"

#ifndef ENVMAP_MAX_MIP_LEVEL
#define ENVMAP_MAX_MIP_LEVEL 8.0
#endif

#ifndef FNC_ENVMAP
#define FNC_ENVMAP

vec3 envMap(vec3 _normal, float _roughness, float _metallic) {

#if defined(SCENE_CUBEMAP)
    float lod = ENVMAP_MAX_MIP_LEVEL * _roughness;
    return textureCube( SCENE_CUBEMAP, _normal, lod).rgb;

#else
    return fakeCube(_normal, toShininess(_roughness, _metallic));

#endif
}

vec3 envMap(vec3 _normal, float _roughness) {
    return envMap(_normal, _roughness, 1.0);
}

#endif