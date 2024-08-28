#include "cast.hlsl"
#include "ao.hlsl"
#include "normal.hlsl"
#include "shading.hlsl"
#include "fog.hlsl"

/*
contributors:  Inigo Quiles
description: Default raymarching renderer
use: 
    - <float4> raymarchDefaultRender( in <float3> rayOriging, in <float3> rayDirection, in <float3> cameraForward)
    - <float4> raymarchDefaultRender( in <float3> rayOriging, in <float3> rayDirection, in <float3> cameraForward, out <float> eyeDepth )
    - <float4> raymarchDefaultRender( in <float3> rayOriging, in <float3> rayDirection, in <float3> cameraForward, out <float> eyeDepth, out <Material> res )
    - <float4> raymarchDefaultRender( in <float3> rayOriging, in <float3> rayDirection, in <float3> cameraForward, out <float3> eyeDepth, out <float3> worldPosition, out <float3> worldNormal ) 
options:
    - RAYMARCH_BACKGROUND: float3(0.0, 0.0, 0.0)
    - RAYMARCH_AOV: return AOVs in a Material structure
*/


#ifndef RAYMARCH_BACKGROUND
#define RAYMARCH_BACKGROUND float3(0.0, 0.0, 0.0)
#endif

#ifndef FNC_RAYMARCH_DEFAULT
#define FNC_RAYMARCH_DEFAULT

float4 raymarchDefaultRender(in float3 rayOrigin, in float3 rayDirection, float3 cameraForward,
                             out float dist, out float eyeDepth, out Material res) { 

    res = raymarchCast(rayOrigin, rayDirection);
    float t = res.sdf;
    float3 worldPos = rayOrigin + t * rayDirection;
    float3 worldNormal = raymarchNormal( worldPos );

    float4 color = float4(RAYMARCH_BACKGROUND, 0.0);
    if (res.valid) {
        res.position = worldPos;
        res.normal = worldNormal;
        res.ambientOcclusion = raymarchAO(res.position, res.normal);
        ShadingData shadingData = shadingDataNew();
        shadingData.V = -rayDirection;
        color = RAYMARCH_SHADING_FNC(res, shadingData);
        dist = t;
    }
    else {
        dist = RAYMARCH_MAX_DIST;
    }
    color.rgb = raymarchFog(color.rgb, t, rayOrigin, rayDirection);

    #if defined(RAYMARCH_AOV)
    // Eye-space depth. See https://www.shadertoy.com/view/4tByz3
    eyeDepth = t * dot(rayDirection, cameraForward);
    #else
    eyeDepth = 0.0;
    #endif

    return color;
}

#endif