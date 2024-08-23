#include "../material.glsl"
#include "../medium.glsl"

/*
contributors:  Inigo Quiles
description: Map of SDF functions to be declare
use: <vec4> raymarchMap( in <vec3> pos ) 
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC raymarchMap
#endif

#ifndef RAYMARCH_VOLUME_MAP_FNC
#define RAYMARCH_VOLUME_MAP_FNC raymarchVolumeMap
#endif

#ifndef FNC_RAYMARCH_MAP
#define FNC_RAYMARCH_MAP

Material RAYMARCH_MAP_FNC( in vec3 pos );
Medium RAYMARCH_VOLUME_MAP_FNC( in vec3 pos );

#endif