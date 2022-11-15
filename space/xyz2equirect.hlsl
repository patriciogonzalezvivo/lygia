#include "../math/const.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: 3D vector to equirect 2D projection 
use: <vec2> xyz2equirect(<vec2> d)
*/

#ifndef FNC_XYZ2EQUIRECT
#define FNC_XYZ2EQUIRECT
float2 xyz2equirect(float3 d) {
    return float2(atan(d.z, d.x) + PI, acos(-d.y)) / float2(TAU, PI);
}
#endif