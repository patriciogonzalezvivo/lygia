#include "cast.hlsl"
#include "ao.hlsl"
#include "normal.hlsl"
#include "softShadow.hlsl"
#include "material.hlsl"
#include "fog.hlsl"

/*
contributors:  Inigo Quiles
description: Default raymarching renderer
use: <vec4> raymarchDefaultRender( in <float3> rayOriging, in <float3> rayDirection, in <float3> cameraForward,
    out <float3> eyeDepth, out <float3> worldPosition, out <float3> worldNormal ) 
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

float4 raymarchDefaultRender(
    in float3 rayOrigin, in float3 rayDirection, float3 cameraForward,
    out float eyeDepth, out float3 worldPos, out float3 worldNormal ) { 

    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    
    RAYMARCHCAST_TYPE res = raymarchCast(rayOrigin, rayDirection);
    float t = res.RAYMARCH_MAP_DISTANCE;

    worldPos = rayOrigin + t * rayDirection;
    worldNormal = raymarchNormal( worldPos );
    color = raymarchMaterial(rayDirection, worldPos, worldNormal, res.RAYMARCH_MAP_MATERIAL);
    color.rgb = raymarchFog(color.rgb, t, rayOrigin, rayDirection);

    // Eye-space depth. See https://www.shadertoy.com/view/4tByz3
    eyeDepth = t * dot(rayDirection, cameraForward);

    return color;
}

#endif