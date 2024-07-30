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
*/

#ifndef FNC_RAYMARCHDEFAULT
#define FNC_RAYMARCHDEFAULT

float4 raymarchDefaultRender(
    in float3 rayOrigin, in float3 rayDirection, float3 cameraForward,
    out float eyeDepth, out float3 worldPos, out float3 worldNormal ) { 

    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    
    Material res = raymarchCast(rayOrigin, rayDirection);
    float t = res.sdf;

    worldPos = rayOrigin + t * rayDirection;
    worldNormal = raymarchNormal( worldPos );
    color = RAYMARCH_MATERIAL_FNC(rayDirection, worldPos, worldNormal, res).albedo;
    color.rgb = raymarchFog(color.rgb, t, rayOrigin, rayDirection);

    // Eye-space depth. See https://www.shadertoy.com/view/4tByz3
    eyeDepth = t * dot(rayDirection, cameraForward);

    return color;
}

#endif