#include "../math/toMat3.wgsl"
#include "../math/const.wgsl"
#include "../space/rotate.wgsl"
#include "../space/lookAtView.wgsl"
#include "raymarch/render.wgsl"
#include "raymarch/volume.wgsl"
#include "material/zero.wgsl"
#include "material/add.wgsl"
#include "material/multiply.wgsl"

/*
contributors:  [Inigo Quiles, Shadi El Hajj, Patricio Gonzalez Vivo]
description: Raymarching template where it needs to define a vec4 raymarchMap( in vec3 pos )
use: 
    - <vec4> raymarch(<mat4> viewMatrix, <vec2> st)
    - <vec4> raymarch(<mat4> viewMatrix, <vec2> st, out <float> eyeDepth)
    - <vec4> raymarch(<mat4> viewMatrix, <vec2> st, out <float> eyeDepth, out <Material> mat)
    - <vec4> raymarch(<vec3> cameraPosition, <vec3> lookAt, <vec2> st)
    - <vec4> raymarch(<vec3> cameraPosition, <vec3> lookAt, <vec2> st, out <float> eyeDepth)
    - <vec4> raymarch(<vec3> cameraPosition, <vec3> lookAt, <vec2> st, out <float> eyeDepth, out <Material> mat)
options:
    - RAYMARCH_RENDER_FNC: default raymarchDefaultRender
    - RAYMARCH_CAMERA_FOV: Filed of view express in degrees. Default 60.0 degrees
    - RAYMARCH_MULTISAMPLE: default 1. If it is greater than 1 it will render multisample
    - RAYMARCH_AOV: return AOVs in a Material structure
    - RAYMARCH_SPHERICAL: will apply a spherical transformation to the coordinates. Useful for rendering fulldome content. FOV should be greater or equal to 180.
examples:
    - /shaders/lighting_raymarching.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define RAYMARCH_RENDER_FNC raymarchDefaultRender

// #define RAYMARCH_VOLUME_RENDER_FNC raymarchVolume

const RAYMARCH_CAMERA_FOV: f32 = 60.0;

let RAYMARCH_MULTISAMPLE_FACTOR = 1.0/float(RAYMARCH_MULTISAMPLE);

fn raymarchModelPosition(st: vec2f) -> vec3f {
    let fov = 1.0 / tan(RAYMARCH_CAMERA_FOV * DEG2RAD * 0.5);
    return normalize(vec3f(st*2.0-1.0, fov));
    let theta = length(st) * RAYMARCH_CAMERA_FOV * DEG2RAD * 0.5;
    let phi = atan(st.y, st.x);
    return vec3f(sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta));
}

fn raymarch(viewMatrix: mat4x4<f32>, st: vec2f, eyeDepth: f32, mat: Material) -> vec4f {

    let camera = viewMatrix[3].xyz;
    let cameraForward = viewMatrix[2].xyz;
    let viewMatrix3 = toMat3(viewMatrix);

    Material matAcc;
    materialZero(matAcc);

    let color = vec4f(0.0, 0.0, 0.0, 0.0);
    eyeDepth = 0.0;

    let pixel = 1.0/RESOLUTION;
    let offset = rotate( vec2f(0.5, 0.0), EIGHTH_PI);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        let rayDirection = viewMatrix3 * raymarchModelPosition(st + offset * pixel);

        let sampleDepth = 0.0;
        let dist = 0.0;

        let opaque = RAYMARCH_RENDER_FNC(camera, rayDirection, cameraForward, dist, sampleDepth, mat);
        color += vec4f(RAYMARCH_VOLUME_RENDER_FNC(camera, rayDirection, st, dist, opaque.rgb), opaque.a);
        color += opaque;

        eyeDepth += sampleDepth;
        
        materialAdd(matAcc, mat, matAcc);

        offset = rotate(offset, HALF_PI);
    }

    eyeDepth *= RAYMARCH_MULTISAMPLE_FACTOR;

        materialMultiply(matAcc, RAYMARCH_MULTISAMPLE_FACTOR, mat);
    
    return color * RAYMARCH_MULTISAMPLE_FACTOR;
    

    let rayDirection = viewMatrix3 * raymarchModelPosition(st);
    let dist = 0.0;

    let opaque = RAYMARCH_RENDER_FNC( camera, rayDirection, cameraForward, dist, eyeDepth, mat);
        let color = vec4f(RAYMARCH_VOLUME_RENDER_FNC(camera, rayDirection, st, dist, opaque.rgb), opaque.a);
        let color = opaque;

    return color;

}

fn raymarcha(viewMatrix: mat4x4<f32>, st: vec2f, eyeDepth: f32) -> vec4f {
    Material mat;
    materialZero(mat);
    return raymarch(viewMatrix, st, eyeDepth, mat);
}

fn raymarchb(viewMatrix: mat4x4<f32>, st: vec2f) -> vec4f {
    let eyeDepth = 0.0;
    return raymarch(viewMatrix, st, eyeDepth);
}

fn raymarch3(cameraPosition: vec3f, cameraLookAt: vec3f, st: vec2f, eyeDepth: f32, mat: Material) -> vec4f {
    let viewMatrix = lookAtView(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st, eyeDepth, mat);
}

fn raymarch3a(cameraPosition: vec3f, cameraLookAt: vec3f, st: vec2f, eyeDepth: f32) -> vec4f {
    let viewMatrix = lookAtView(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st, eyeDepth);
}

fn raymarch3b(cameraPosition: vec3f, cameraLookAt: vec3f, st: vec2f) -> vec4f {
    let viewMatrix = lookAtView(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st);
}
