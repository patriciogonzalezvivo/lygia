#include "../math/toMat3.glsl"
#include "../math/const.glsl"
#include "../space/rotate.glsl"
#include "../space/lookAtView.glsl"
#include "raymarch/render.glsl"
#include "raymarch/volume.glsl"
#include "material/zero.glsl"
#include "material/add.glsl"
#include "material/multiply.glsl"

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
examples:
    - /shaders/lighting_raymarching.frag
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

#ifndef RAYMARCH_CAMERA_FOV
#define RAYMARCH_CAMERA_FOV 60.0
#endif

#ifndef FNC_RAYMARCH
#define FNC_RAYMARCH

#if defined(RAYMARCH_MULTISAMPLE)
const float RAYMARCH_MULTISAMPLE_FACTOR = 1.0/float(RAYMARCH_MULTISAMPLE);
#endif

vec4 raymarch(mat4 viewMatrix, vec2 st, out float eyeDepth, out Material mat) {

    float fov = 1.0 / tan(RAYMARCH_CAMERA_FOV * DEG2RAD * 0.5);
    vec3 camera = viewMatrix[3].xyz;
    vec3 cameraForward = viewMatrix[2].xyz;
    mat3 viewMatrix3 = toMat3(viewMatrix);

#if defined(RAYMARCH_MULTISAMPLE)

    #if defined(RAYMARCH_AOV)
    Material matAcc;
    materialZero(matAcc);
    #endif

    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    eyeDepth = 0.0;

    vec2 pixel = 1.0/RESOLUTION;
    vec2 offset = rotate( vec2(0.5, 0.0), EIGHTH_PI);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        vec3 rayDirection = viewMatrix3 * normalize(vec3((st + offset * pixel)*2.0-1.0, fov));

        float sampleDepth = 0.0;
        float dist = 0.0;

        vec4 opaque = RAYMARCH_RENDER_FNC(camera, rayDirection, cameraForward, dist, sampleDepth, mat);
        #ifdef RAYMARCH_VOLUME
        color += vec4(RAYMARCH_VOLUME_RENDER_FNC(camera, rayDirection, st, dist, opaque.rgb), opaque.a);
        #else
        color += opaque;
        #endif

        eyeDepth += sampleDepth;
        
        #if defined(RAYMARCH_AOV)
        materialAdd(matAcc, mat, matAcc);
        #endif

        offset = rotate(offset, HALF_PI);
    }

    eyeDepth *= RAYMARCH_MULTISAMPLE_FACTOR;

    #if defined(RAYMARCH_AOV)  
        materialMultiply(matAcc, RAYMARCH_MULTISAMPLE_FACTOR, mat);
    #endif
    
    return color * RAYMARCH_MULTISAMPLE_FACTOR;
    
#else // Single sample

    vec3 rayDirection = viewMatrix3 * normalize(vec3(st*2.0-1.0, fov));
    float dist = 0.0;

    vec4 opaque = RAYMARCH_RENDER_FNC( camera, rayDirection, cameraForward, dist, eyeDepth, mat);
    #ifdef RAYMARCH_VOLUME
        vec4 color = vec4(RAYMARCH_VOLUME_RENDER_FNC(camera, rayDirection, st, dist, opaque.rgb), opaque.a);
    #else
        vec4 color = opaque;
    #endif

    return color;

#endif
}

vec4 raymarch(mat4 viewMatrix, vec2 st, out float eyeDepth) {
    Material mat;
    materialZero(mat);
    return raymarch(viewMatrix, st, eyeDepth, mat);
}

vec4 raymarch(mat4 viewMatrix, vec2 st) {
    float eyeDepth = 0.0;
    return raymarch(viewMatrix, st, eyeDepth);
}

vec4 raymarch(vec3 cameraPosition, vec3 cameraLookAt, vec2 st, out float eyeDepth, out Material mat) {
    mat4 viewMatrix = lookAtView(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st, eyeDepth, mat);
}

vec4 raymarch(vec3 cameraPosition, vec3 cameraLookAt, vec2 st, out float eyeDepth) {
    mat4 viewMatrix = lookAtView(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st, eyeDepth);
}

vec4 raymarch(vec3 cameraPosition, vec3 cameraLookAt, vec2 st) {
    mat4 viewMatrix = lookAtView(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st);
}

#endif