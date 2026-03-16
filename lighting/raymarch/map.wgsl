#include "../material.wgsl"
#include "../medium.wgsl"

/*
contributors:  Inigo Quiles
description: Map of SDF functions to be declare
use: <vec4> raymarchMap( in <vec3> pos ) 
examples:
    - /shaders/lighting_raymarching.frag
*/

// #define RAYMARCH_MAP_FNC raymarchMap

// #define RAYMARCH_VOLUME_MAP_FNC raymarchVolumeMap

Material RAYMARCH_MAP_FNC( in vec3 pos );
Medium RAYMARCH_VOLUME_MAP_FNC( in vec3 pos );
