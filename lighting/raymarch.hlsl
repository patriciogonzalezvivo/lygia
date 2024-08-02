/*
contributors: Patricio Gonzalez Vivo
description: Raymarching template where it needs to define a float4 raymarchMSap( in float3 pos )
use: <float4> raymarch(<float3> cameraPosition, <float3> lookAt, <float2> st,
    out <float3> eyeDepth, out <float3> worldPosition, out <float3> worldNormal )
options:
    - RAYMARCH_CAMERA_MATRIX_FNC(RO, TA): default raymarchCamera(RO, TA)
    - RAYMARCH_RENDER_FNC(RO, RD): default raymarchDefaultRender(RO, RD, TA)
    - RAYMARCH_CAMERA_FOV
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RAYMARCH_CAMERA_MATRIX_FNC
#define RAYMARCH_CAMERA_MATRIX_FNC raymarchCamera
#endif

#ifndef RAYMARCH_RENDER_FNC
#define RAYMARCH_RENDER_FNC raymarchDefaultRender
#endif

#ifndef RESOLUTION
#define RESOLUTION _ScreenParams.xy
#endif

#ifndef RAYMARCH_CAMERA_FOV
#define RAYMARCH_CAMERA_FOV 60.0
#endif

#include "../math/const.hlsl"
#include "../space/rotate.hlsl"
#include "raymarch/render.hlsl"
#include "raymarch/camera.hlsl"

#ifndef ENG_RAYMARCH
#define ENG_RAYMARCH

float4 raymarch( float3 camera, float3 ta, float2 st,
    out float eyeDepth, out float3 worldPos, out float3 worldNormal) {

    float3x3 ca = RAYMARCH_CAMERA_MATRIX_FNC(camera, ta);
    float fov = 1.0/tan(RAYMARCH_CAMERA_FOV*PI/180.0/2.0);
    float3 cameraForward = normalize(ta - camera);
    st.x = 1.0 - st.x;

#if defined(RAYMARCH_MULTISAMPLE)
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    eyeDepth = 0.0;
    worldPos = float3(0.0,0.0, 0.0);
    worldNormal = float3(0.0, 0.0, 0.0);
    float2 pixel = 1.0/RESOLUTION;
    float2 offset = rotate( float2(0.5, 0.0), HALF_PI/4.);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        float3 rd = mul(ca, normalize(float3((st + offset * pixel)*2.0-1.0, fov)));
        float sampleDepth = 0.0;
        float3 sampleWorldPos = float3(0.0, 0.0, 0.0);
        float3 sampleWorldNormal = float3(0.0, 0.0, 0.0);
        color += RAYMARCH_RENDER_FNC( camera, rd, cameraForward,
            sampleDepth, sampleWorldPos, sampleWorldNormal );
        eyeDepth += sampleDepth;
        worldPos += sampleWorldPos;
        worldNormal += sampleWorldNormal;
        offset = rotate(offset, HALF_PI);
    }
    eyeDepth /= float(RAYMARCH_MULTISAMPLE);
    worldPos /= float(RAYMARCH_MULTISAMPLE);
    worldNormal /= float(RAYMARCH_MULTISAMPLE);
    return color/float(RAYMARCH_MULTISAMPLE);
#else
    float3 rd = mul(ca, normalize(float3(st * 2.0 - 1.0, fov)));
    return RAYMARCH_RENDER_FNC( camera, rd, cameraForward, eyeDepth, worldPos, worldNormal );
#endif
}

float4 raymarch(float3 camera, float3 ta, float2 st) {
    float depth = 0.0;
    float3 worldPos = float3(0.0, 0.0, 0.0);
    float3 worldNormal = float3(0.0, 0.0, 0.0);
    return raymarch(camera, ta, st, depth, worldPos, worldNormal);
}

float4 raymarch(float3 camera, float2 st) {
    return raymarch(camera, float3(0.0, 0.0, 0.0), st);
}

#endif