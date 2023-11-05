#include "../space/depth2viewZ.glsl"
#include "../space/screen2viewPosition.glsl"

#include "../sample.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: sampler the view Positiong from depthmap texture 
use: <vec4> sampleViewPosition(<SAMPLER_TYPE> texDepth, <vec2> st [, <float> near, <float> far])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - CAMERA_NEAR_CLIP: required
    - CAMERA_FAR_CLIP: required
    - CAMERA_ORTHOGRAPHIC_PROJECTION, if it's not present is consider a PERECPECTIVE camera
    - CAMERA_PROJECTION_MATRIX: mat4 matrix with camera projection
    - INVERSE_CAMERA_PROJECTION_MATRIX: mat4 matrix with the inverse camara projection
*/

#ifndef FNC_SAMPLEVIEWPOSITION
#define FNC_SAMPLEVIEWPOSITION
vec4 sampleViewPosition(const in float depth, const in vec2 st, const in float near, const in float far) {
    float viewZ = depth2viewZ(depth, near, far);
    return screen2viewPosition(st, depth, viewZ);
}

vec4 sampleViewPosition(SAMPLER_TYPE texDepth, const in vec2 st, const in float near, const in float far) {
    float depth = SAMPLER_FNC(texDepth, st).r;
    float viewZ = depth2viewZ(depth, near, far);
    return screen2viewPosition(st, depth, viewZ);
}

#if defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
vec4 sampleViewPosition(SAMPLER_TYPE texDepth, const in vec2 st) {
    return sampleViewPosition( texDepth, st, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP); 
}
#endif

#endif
