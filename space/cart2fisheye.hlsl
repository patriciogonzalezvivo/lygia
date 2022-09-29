#include "../math/const.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: UV to equirect 3D fisheye vector 
use: <float3> cart2equirect(<float2> uv, <float> MinCos)
*/

#ifndef FNC_CART2FISHEYE
#define FNC_CART2FISHEYE
float3 cart2fisheye(float2 uv) {
    float2 ndc = uv * 2.0 - 1.0;
    // ndc.x *= u_resolution.x / u_resolution.y;
    float R = sqrt(ndc.x * ndc.x + ndc.y * ndc.y);
    float3 dir = float3(ndc.x / R, 0.0, ndc.y / R);
    float Phi = (R) * PI * 0.52;
    dir.y   = cos(Phi);//clamp(, MinCos, 1.0);
    dir.xz *= sqrt(1.0 - dir.y * dir.y);
    return dir;
}
#endif