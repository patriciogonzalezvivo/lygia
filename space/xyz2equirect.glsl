#include "../math/const.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 3D vector to equirect 2D projection
use: <vec2> xyz2equirect(<vec2> d)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_XYZ2EQUIRECT
#define FNC_XYZ2EQUIRECT
vec2 xyz2equirect(vec3 d) {
    return vec2(atan(d.z, d.x) + PI, acos(-d.y)) / vec2(2.0 * PI, PI);
}
#endif