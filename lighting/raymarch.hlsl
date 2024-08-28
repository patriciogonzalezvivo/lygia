#include "../math/const.hlsl"
#include "../space/rotate.hlsl"
#include "../space/lookAtView.hlsl"
#include "raymarch/render.hlsl"
#include "raymarch/volume.hlsl"
#include "material/zero.hlsl"

/*
contributors:  [Inigo Quiles, Shadi El Hajj, Patricio Gonzalez Vivo]
description: Raymarching template where it needs to define a float4 raymarchMSap( in float3 pos )
use: 
    - <float4> raymarch(<float4x4> viewMatrix, <float2> st)
    - <float4> raymarch(<float4x4> viewMatrix, <float2> st, out <float> eyeDepth)
    - <float4> raymarch(<float4x4> viewMatrix, <float2> st, out <float> eyeDepth, out <Material> mat)
    - <float4> raymarch(<float3> cameraPosition, <float3> lookAt, <float2> st)
    - <float4> raymarch(<float3> cameraPosition, <float3> lookAt, <float2> st, out <float> eyeDepth)
    - <float4> raymarch(<float3> cameraPosition, <float3> lookAt, <float2> st, out <float> eyeDepth, out <Material> mat)
options:
    - RAYMARCH_RENDER_FNC: default raymarchDefaultRender
    - RAYMARCH_CAMERA_FOV: Filed of view express in degrees. Default 60.0 degrees
    - RAYMARCH_MULTISAMPLE: default 1. If it is greater than 1 it will render multisample
    - RAYMARCH_AOV: return AOVs in a Material structure
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RAYMARCH_RENDER_FNC
#define RAYMARCH_RENDER_FNC raymarchDefaultRender
#endif

#ifndef RAYMARCH_VOLUME_RENDER_FNC
#define RAYMARCH_VOLUME_RENDER_FNC raymarchVolume
#endif

#ifndef RESOLUTION
#define RESOLUTION _ScreenParams.xy
#endif

#ifndef RAYMARCH_CAMERA_FOV
#define RAYMARCH_CAMERA_FOV 60.0
#endif

#ifndef FNC_RAYMARCH
#define FNC_RAYMARCH

#if defined(RAYMARCH_MULTISAMPLE)
static const float RAYMARCH_MULTISAMPLE_FACTOR = 1.0/float(RAYMARCH_MULTISAMPLE);
#endif

float4 raymarch(float4x4 viewMatrix, float2 st, out float eyeDepth, out Material mat) {

    float fov = 1.0 / tan(RAYMARCH_CAMERA_FOV * DEG2RAD * 0.5);
    float3 camera = float3(viewMatrix._m03, viewMatrix._m13, viewMatrix._m23);
    float3 cameraForward = float3(viewMatrix._m02, viewMatrix._m12, viewMatrix._m22);

#if defined(RAYMARCH_MULTISAMPLE)

    #if defined(RAYMARCH_AOV)
    Material matAcc;
    materialZero(matAcc);
    #endif

    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    eyeDepth = 0.0;

    float2 pixel = 1.0/RESOLUTION;
    float2 offset = rotate( float2(0.5, 0.0), EIGHTH_PI);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        float3 rayDirection = mul((float3x3)viewMatrix, normalize(float3((st + offset * pixel)*2.0-1.0, fov)));
        
        float sampleDepth = 0.0;
        float dist = 0.0;

        float4 opaque = RAYMARCH_RENDER_FNC(camera, rayDirection, cameraForward, dist, sampleDepth, mat);
        #ifdef RAYMARCH_VOLUME
        color += float4(RAYMARCH_VOLUME_RENDER_FNC(camera, rayDirection, st, dist, opaque.rgb), opaque.a);
        #else
        color += opaque;
        #endif

        eyeDepth += sampleDepth;

        #if defined(RAYMARCH_AOV)
            // Accumulate material properties
            materialAdd(matAcc, mat, matAcc);
        #endif

        offset = rotate(offset, HALF_PI);
    }

    eyeDepth *= RAYMARCH_MULTISAMPLE_FACTOR;

    #if defined(RAYMARCH_AOV)
        // Average material
        materialMultiply(matAcc, RAYMARCH_MULTISAMPLE_FACTOR, mat);
    #endif
    
    return color * RAYMARCH_MULTISAMPLE_FACTOR;

#else // Single sample

    float3 rayDirection = mul((float3x3)viewMatrix, normalize(float3(st * 2.0 - 1.0, fov)));
    float dist = 0.0;

    float4 opaque = RAYMARCH_RENDER_FNC(camera, rayDirection, cameraForward, dist, eyeDepth, mat);
    #ifdef RAYMARCH_VOLUME
        float4 color = float4(RAYMARCH_VOLUME_RENDER_FNC(camera, rayDirection, st, dist, opaque.rgb), opaque.a);
    #else
        float4 color = opaque;
    #endif

    return color;

#endif
}

float4 raymarch(float4x4 viewMatrix, float2 st, out float eyeDepth) {
    Material mat;
    materialZero(mat);
    return raymarch(viewMatrix, st, eyeDepth, mat);
}

float4 raymarch(float4x4 viewMatrix, float2 st) {
    float eyeDepth = 0.0;
    return raymarch(viewMatrix, st, eyeDepth);
}

float4 raymarch(float3 cameraPosition, float3 cameraLookAt, float2 st, out float eyeDepth, out Material mat) {
    float4x4 viewMatrix = lookAtView(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st, eyeDepth, mat);
}

float4 raymarch(float3 cameraPosition, float3 cameraLookAt, float2 st, out float eyeDepth) {
    float4x4 viewMatrix = lookAtView(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st, eyeDepth);
}

float4 raymarch(float3 cameraPosition, float3 cameraLookAt, float2 st) {
    float4x4 viewMatrix = lookAtView(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st);
}

#endif
