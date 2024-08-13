#include "cast.hlsl"
#include "ao.hlsl"
#include "normal.hlsl"
#include "shading.hlsl"
#include "fog.hlsl"

/*
contributors:  Inigo Quiles
description: Default raymarching renderer
use: <float4> raymarchDefaultRender( in <float3> rayOriging, in <float3> rayDirection, in <float3> cameraForward,
    out <float3> eyeDepth, out <float3> worldPosition, out <float3> worldNormal ) 
options:
    - RAYMARCH_BACKGROUND: float3(0.0)
*/

#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND float3(0.0, 0.0, 0.0)
#endif

#ifndef FNC_RAYMARCH_DEFAULT
#define FNC_RAYMARCH_DEFAULT

float4 raymarchDefaultRender(
    in float3 rayOrigin, in float3 rayDirection, float3 cameraForward,
    out float eyeDepth, out float3 worldPos, out float3 worldNormal ) { 

    Material res = raymarchCast(rayOrigin, rayDirection);
    float t = res.sdf;

    worldPos = rayOrigin + t * rayDirection;
    worldNormal = raymarchNormal( worldPos );

    float4 color = float4(RAYMARCH_BACKGROUND, 0.0);
    if (res.valid) {
        res.position = worldPos;
        res.normal = worldNormal;
        res.V = -rayDirection;
        color = RAYMARCH_SHADING_FNC(res);
    }
    color.rgb = raymarchFog(color.rgb, t, rayOrigin, rayDirection);

    // Eye-space depth. See https://www.shadertoy.com/view/4tByz3
    eyeDepth = t * dot(rayDirection, cameraForward);

    return color;
}

#endif