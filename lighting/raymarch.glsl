/*
original_author: Patricio Gonzalez Vivo
description: raymarching template where it needs to define a vec4 raymarchMSap( in vec3 pos ) 
use: <vec4> raymarch(<vec3> camera, <vec2> st)
options:
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR:     in glslViewer is u_lightColor
    - RAYMARCH_AMBIENT: defualt vec3(1.0)
    - RAYMARCH_BACKGROUND: default vec3(0.0)
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

#include "raymarch/default.glsl"
#include "raymarch/camera.glsl"

vec4 raymarch(vec3 camera, vec3 ta, vec2 st) {
    mat3 ca = RAYMARCH_CAMERA_MATRIX_FNC(camera, ta);
    vec3 rd = ca * normalize(vec3(st*2.0-1.0, 3.0));
    
    return RAYMARCH_RENDER_FNC( camera * 0.11, rd );
}


vec4 raymarch(vec3 camera, vec2 st) {
    return raymarch(camera, vec3(0.0), st);
}

#endif