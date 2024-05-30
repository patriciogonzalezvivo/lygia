#include "cast.hlsl"
#include "ao.hlsl"
#include "normal.hlsl"
#include "softShadow.hlsl"
#include "material.hlsl"

/*
contributors:  Inigo Quiles
description: Default raymarching renderer
use: <float4> raymarchDefaultRender( in <float3> ro, in <float3> rd ) 
options:
    - LIGHT_COLOR: float3(0.5) or u_lightColor in glslViewer
    - LIGHT_POSITION: float3(0.0, 10.0, -50.0) or u_light in GlslViewer
    - LIGHT_DIRECTION;
    - RAYMARCH_BACKGROUND: float3(0.0)
    - RAYMARCH_AMBIENT: float3(1.0)
    - RAYMARCH_MATERIAL_FNC raymarchDefaultMaterial
*/

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE a
#endif

#ifndef RAYMARCH_MAP_MATERIAL
#define RAYMARCH_MAP_MATERIAL rgb
#endif

#ifndef RAYMARCH_MAP_MATERIAL_TYPE
#define RAYMARCH_MAP_MATERIAL_TYPE float3
#endif

#ifndef RAYMARCHCAST_TYPE
#define RAYMARCHCAST_TYPE float4
#endif

#ifndef FNC_RAYMARCHDEFAULT
#define FNC_RAYMARCHDEFAULT

float4 raymarchDefaultRender( in float3 ray_origin, in float3 ray_direction ) { 
    float3 col = float3(0.0, 0.0, 0.0);
    
    RAYMARCHCAST_TYPE res = raymarchCast(ray_origin, ray_direction);
    float t = res.RAYMARCH_MAP_DISTANCE;

    float3 pos = ray_origin + t * ray_direction;
    float3 nor = raymarchNormal( pos );
    col = raymarchMaterial(ray_direction, pos, nor, res.RAYMARCH_MAP_MATERIAL);

    return float4( saturate(col), t );
}

#endif