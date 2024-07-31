/*
contributors: Patricio Gonzalez Vivo
description: Raymarching template where it needs to define a vec4 raymarchMap( in vec3 pos )
use: <vec4> raymarch(<vec3> cameraPosition, <vec3> lookAt, <vec2> st,
    out <vec3> eyeDepth, out <vec3> worldPosition, out <vec3> worldNormal )
options:
    - RAYMARCH_CAMERA_MATRIX_FNC(RO, TA): default raymarchCamera(RO, TA)
    - RAYMARCH_RENDER_FNC(RO, RD): default raymarchDefaultRender(RO, RD, TA)
    - RAYMARCH_CAMERA_FOV
examples:
    - /shaders/lighting_raymarching.frag
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

#ifndef RAYMARCH_CAMERA_FOV
#define RAYMARCH_CAMERA_FOV 60.0
#endif

#include "../math/const.glsl"
#include "../space/rotate.glsl"
#include "raymarch/render.glsl"
#include "raymarch/camera.glsl"

#ifndef ENG_RAYMARCHING
#define ENG_RAYMARCHING

vec4 raymarch( vec3 camera, vec3 ta, vec2 st,
    out float eyeDepth, out vec3 worldPos, out vec3 worldNormal) {

    mat3 ca = RAYMARCH_CAMERA_MATRIX_FNC(camera, ta);
    float fov = 1.0/tan(RAYMARCH_CAMERA_FOV*PI/180.0/2.0);
    vec3 cameraForward = normalize(ta - camera);
    st.x = 1.0 - st.x;

#if defined(RAYMARCH_MULTISAMPLE)
    vec4 color = vec4(0.0);
    eyeDepth = 0.0;
    worldPos = vec3(0.0);
    worldNormal = vec3(0.0);
    vec2 pixel = 1.0/RESOLUTION;
    vec2 offset = rotate( vec2(0.5, 0.0), HALF_PI/4.);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        vec3 rd = ca * normalize(vec3((st + offset * pixel)*2.0-1.0, fov));
        float sampleDepth = 0.0;
        vec3 sampleWorldPos = vec3(0.0);
        vec3 sampleWorldNormal = vec3(0.0);
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
    vec3 rd = ca * normalize(vec3(st*2.0-1.0, fov));
    return RAYMARCH_RENDER_FNC( camera, rd, cameraForward, eyeDepth, worldPos, worldNormal );
#endif
}

vec4 raymarch(vec3 camera, vec3 ta, vec2 st) {
    float depth = 0.0;
    vec3 worldPos = vec3(0.0);
    vec3 worldNormal = vec3(0.0);
    return raymarch(camera, ta, st, depth, worldPos, worldNormal);
}

vec4 raymarch(vec3 camera, vec2 st) {
    return raymarch(camera, vec3(0.0), st);
}

#endif