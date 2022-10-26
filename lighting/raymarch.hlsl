/*
original_author: Patricio Gonzalez Vivo
description: raymarching template where it needs to define a float4 raymarchMSap( in float3 pos ) 
use: <float4> raymarch(<float3> camera, <float2> st)
options:
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR:     in glslViewer is u_lightColor
    - RAYMARCH_AMBIENT: defualt float3(1.0, 1.0, 1.0)
    - RAYMARCH_BACKGROUND: default float3(0.0, 0.0, 0.0)
    - RAYMARCH_CAMERA_MATRIX_FNC(RO, TA): default raymarchCamera(RO, TA)
    - RAYMARCH_RENDER_FNC(RO, RD): default raymarchDefaultRender(RO, RD)  
*/

#ifndef RAYMARCH_CAMERA_MATRIX_FNC
#define RAYMARCH_CAMERA_MATRIX_FNC raymarchCamera
#endif

#ifndef RAYMARCH_RENDER_FNC
#define RAYMARCH_RENDER_FNC raymarchDefaultRender
#endif

#ifndef ENG_RAYMARCHING
#define ENG_RAYMARCHING

#include "raymarch/default.hlsl"
#include "raymarch/camera.hlsl"

float4 raymarch(float3 camera, float3 ta, float2 st) {
    float3x3 ca = RAYMARCH_CAMERA_MATRIX_FNC(camera, ta);
    float3 rd = mul(ca, normalize(float3(st*2.0-1.0, 3.0)));
    
    return RAYMARCH_RENDER_FNC( camera * 0.11, rd );
}


float4 raymarch(float3 camera, float2 st) {
    return raymarch(camera, float3(0.0, 0.0, 0.0), st);
}

#endif