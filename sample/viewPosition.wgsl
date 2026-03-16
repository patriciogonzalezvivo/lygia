#include "../space/depth2viewZ.wgsl"
#include "../space/screen2viewPosition.wgsl"

#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample the view Position from depthmap texture
use: <vec4> sampleViewPosition(<SAMPLER_TYPE> texDepth, <vec2> st [, <float> near, <float> far])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - CAMERA_NEAR_CLIP: required
    - CAMERA_FAR_CLIP: required
    - CAMERA_ORTHOGRAPHIC_PROJECTION, if it's not present is consider a PERECPECTIVE camera
    - CAMERA_PROJECTION_MATRIX: mat4 matrix with camera projection
    - INVERSE_CAMERA_PROJECTION_MATRIX: mat4 matrix with the inverse camara projection
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleViewPosition(depth: f32, st: vec2f, near: f32, far: f32) -> vec4f {
    let viewZ = depth2viewZ(depth, near, far);
    return screen2viewPosition(st, depth, viewZ);
}

fn sampleViewPositiona(texDepth: SAMPLER_TYPE, st: vec2f, near: f32, far: f32) -> vec4f {
    let depth = SAMPLER_FNC(texDepth, st).r;
    let viewZ = depth2viewZ(depth, near, far);
    return screen2viewPosition(st, depth, viewZ);
}

fn sampleViewPositionb(texDepth: SAMPLER_TYPE, st: vec2f) -> vec4f {
    return sampleViewPosition( texDepth, st, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP); 
}
