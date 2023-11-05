#include "../math/const.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: fisheye 2D projection to 3D vector 
use: <vec3> fisheye2xyz(<vec2> uv)
*/

#ifndef FNC_FISHEYE2XYZ
#define FNC_FISHEYE2XYZ
vec3 fisheye2xyz(vec2 uv) {
    vec2 ndc = uv * 2.0 - 1.0;
    float R = sqrt(ndc.x * ndc.x + ndc.y * ndc.y);
    vec3 dir = vec3(ndc.x / R, 0.0, ndc.y / R);
    float Phi = (R) * PI * 0.52;
    dir.y   = cos(Phi);//clamp(, MinCos, 1.0);
    dir.xz *= sqrt(1.0 - dir.y * dir.y);
    return dir;
}
#endif