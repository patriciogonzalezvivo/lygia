#include "../math/const.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: UV to equirect 3D fisheye vector 
use: <vec3> cart2equirect(<vec2> uv, <float> MinCos)
*/

#ifndef FNC_CART2FISHEYE
#define FNC_CART2FISHEYE
vec3 cart2fisheye(vec2 uv) {
    vec2 ndc = uv * 2.0 - 1.0;
    // ndc.x *= u_resolution.x / u_resolution.y;
    float R = sqrt(ndc.x * ndc.x + ndc.y * ndc.y);
    vec3 dir = vec3(ndc.x / R, 0.0, ndc.y / R);
    float Phi = (R) * PI * 0.52;
    dir.y   = cos(Phi);//clamp(, MinCos, 1.0);
    dir.xz *= sqrt(1.0 - dir.y * dir.y);
    return dir;
}
#endif