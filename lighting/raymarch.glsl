/*
contributors: Patricio Gonzalez Vivo
description: Raymarching template where it needs to define a vec4 raymarchMap( in vec3 pos )
use: <vec4> raymarch(<vec3> camera, <vec2> st)
options:
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_DIRECTION
    - LIGHT_COLOR: in glslViewer is u_lightColor
    - RAYMARCH_AMBIENT: defualt vec3(1.0)
    - RAYMARCH_MULTISAMPLE: null
    - RAYMARCH_BACKGROUND: default vec3(0.0)
    - RAYMARCH_CAMERA_MATRIX_FNC(RO, TA): default raymarchCamera(RO, TA)
    - RAYMARCH_RENDER_FNC(RO, RD): default raymarchDefaultRender(RO, RD)
    - RESOLUTION: nan
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
#define RAYMARCH_CAMERA_FOV 3.0
#endif

#ifndef RAYMARCH_CAMERA_SCALE
#define RAYMARCH_CAMERA_SCALE 0.11
#endif

#include "../math/const.glsl"
#include "../space/rotate.glsl"
#include "raymarch/render.glsl"
#include "raymarch/camera.glsl"

#ifndef ENG_RAYMARCHING
#define ENG_RAYMARCHING
vec4 raymarch(vec3 camera, vec3 ta, vec2 st) {
    mat3 ca = RAYMARCH_CAMERA_MATRIX_FNC(camera, ta);
    
#if defined(RAYMARCH_MULTISAMPLE)
    vec4 color = vec4(0.0);
    vec2 pixel = 1.0/RESOLUTION;
    vec2 offset = rotate( vec2(0.5, 0.0), HALF_PI/4.);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        vec3 rd = ca * normalize(vec3((st + offset * pixel)*2.0-1.0, RAYMARCH_CAMERA_FOV));
        color += RAYMARCH_RENDER_FNC( camera * RAYMARCH_CAMERA_SCALE, rd);
        offset = rotate(offset, HALF_PI);
    }
    return color/float(RAYMARCH_MULTISAMPLE);
#else
    vec3 rd = ca * normalize(vec3(st*2.0-1.0, RAYMARCH_CAMERA_FOV));
    return RAYMARCH_RENDER_FNC( camera * RAYMARCH_CAMERA_SCALE, rd);
#endif
}

vec4 raymarch(vec3 camera, vec2 st) {
    return raymarch(camera, vec3(0.0), st);
}

#endif