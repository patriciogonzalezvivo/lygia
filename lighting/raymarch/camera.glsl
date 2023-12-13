#include "../../space/lookAt.glsl"

/*
contributors:  Inigo Quiles
description: set a camera for raymarching 
use: <mat3> raymarchCamera(in <vec3> ro, [in <vec3> ta [, in <vec3> up] ])
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef FNC_RAYMARCHCAMERA
#define FNC_RAYMARCHCAMERA

mat3 raymarchCamera( in vec3 ro, in vec3 ta, in vec3 up ) {
    vec3 cw = normalize(ta-ro);
    vec3 cu = normalize( cross(cw,up) );
    vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

mat3 raymarchCamera( in vec3 ro, in vec3 ta, float cr ) {
    vec3 cw = normalize(ta-ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv =          ( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

mat3 raymarchCamera( in vec3 ro, in vec3 ta ) {
    return raymarchCamera( ro, ta, vec3(0.0, 1.0, 0.0) );
}

mat3 raymarchCamera( in vec3 ro ) {
    return raymarchCamera( ro, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0) );
}

#endif