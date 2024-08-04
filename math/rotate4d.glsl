/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <mat4> rotate4d(<vec3> axis, <float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE4D
#define FNC_ROTATE4D
mat4 rotate4d(in vec3 a, const in float r) {
    a = normalize(a);
    float s = sin(r);
    float c = cos(r);
    float oc = 1.0 - c;
    vec4 col1 = vec4(oc * a.x * a.x + c, oc * a.x * a.y + a.z * s, oc * a.z * a.x - a.y * s, 0.0);
    vec4 col2 = vec4(oc * a.x * a.y - a.z * s, oc * a.y * a.y + c, oc * a.y * a.z + a.x * s, 0.0);
    vec4 col3 = vec4(oc * a.z * a.x + a.y * s, oc * a.y * a.z - a.x * s, oc * a.z * a.z + c, 0.0);
    vec4 col4 = vec4(0.0, 0.0, 0.0, 1.0);
    return mat4(col1, col2, col3, col4);
}
#endif
