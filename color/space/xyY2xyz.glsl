/*
contributors: Patricio Gonzalez Vivo
description: 'Converts from xyY to XYZ'
use: <vec3|vec4> xyY2xyz(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_XYY2XYZ
#define FNC_XYY2XYZ
vec3 xyY2xyz(const in vec3 xyY) {
    float Y = xyY.z;
    float f = 1.0/xyY.y;
    float x = Y * xyY.x * f;
    float z = Y * (1.0 - xyY.x - xyY.y) * f;
    return vec3(x, Y, z);
}
vec4 xyY2xyz(const in vec4 xyY) { return vec4(xyY2xyz(xyY.xyz), xyY.a); }
#endif