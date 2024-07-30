#include "../material.hlsl"

/*
contributors:  Inigo Quiles
description: Map of SDF functions to be declare
use: <float4> raymarchMap( in <float3> pos ) 
*/

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC raymarchMap
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE a
#endif

#ifndef RAYMARCH_MAP_MATERIAL
#define RAYMARCH_MAP_MATERIAL rgb
#endif

#ifndef FNC_RAYMARCHMAP
#define FNC_RAYMARCHMAP

Material RAYMARCH_MAP_FNC( in float3 pos );

#endif