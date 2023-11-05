/*
contributors:  Inigo Quiles
description: map of SDF functions to be declare
use: <float4> raymarchMap( in <float3> pos ) 
*/

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_TYPE
#define RAYMARCH_MAP_TYPE float4
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE a
#endif

#ifndef RAYMARCH_MAP_MATERIAL
#define RAYMARCH_MAP_MATERIAL rgb
#endif

#ifndef FNC_RAYMARCHMAP
#define FNC_RAYMARCHMAP

RAYMARCH_MAP_TYPE RAYMARCH_MAP_FNC( in float3 pos );

#endif