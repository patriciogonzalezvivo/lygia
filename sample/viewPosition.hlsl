#include "../space/depth2viewZ.hlsl"
#include "../space/screen2viewPosition.hlsl"
#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample the view Position from depthmap texture
use: <float4> sampleViewPosition(<SAMPLER_TYPE> texDepth, <float2> st [, <float> near, <float> far])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - CAMERA_NEAR_CLIP: required
    - CAMERA_FAR_CLIP: required
    - CAMERA_ORTHOGRAPHIC_PROJECTION, if it's not present is consider a PERECPECTIVE camera
    - CAMERA_PROJECTION_MATRIX: float4x4 matrix with camera projection
    - INVERSE_CAMERA_PROJECTION_MATRIX: float4x4 matrix with the inverse camara projection
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLEVIEWPOSITION
#define FNC_SAMPLEVIEWPOSITION
float4 sampleViewPosition(const in float depth, const in float2 st, const in float near, const in float far) {
    float viewZ = depth2viewZ(depth, near, far);
    return screen2viewPosition(st, depth, viewZ);
}

float4 sampleViewPosition(SAMPLER_TYPE texDepth, const in float2 st, const in float near, const in float far) {
    float depth = SAMPLER_FNC(texDepth, st).r;
    float viewZ = depth2viewZ(depth, near, far);
    return screen2viewPosition(st, depth, viewZ);
}

#if defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
float4 sampleViewPosition(SAMPLER_TYPE texDepth, const in float2 st) {
    return sampleViewPosition( texDepth, st, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP); 
}
#endif

#endif
