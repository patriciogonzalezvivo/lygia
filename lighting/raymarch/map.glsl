#include "../material.glsl"

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

#ifndef FNC_RAYMARCHMAP
#define FNC_RAYMARCHMAP

Material RAYMARCH_MAP_FNC( in vec3 pos );

#endif