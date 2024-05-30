/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <mat4> rotate4d(<vec3> axis, <float> radians)
*/

#ifndef FNC_ROTATE4D
#define FNC_ROTATE4D
mat4 rotate4d(in vec3 a, const in float r) {
    a = normalize(a);
    float s = sin(r);
    float c = cos(r);
    float oc = 1.0 - c;
    return mat4(oc * a.x * a.x + c,         oc * a.x * a.y - a.z * s,   oc * a.z * a.x + a.y * s,   0.0,
                oc * a.x * a.y + a.z * s,   oc * a.y * a.y + c,         oc * a.y * a.z - a.x * s,   0.0,
                oc * a.z * a.x - a.y * s,   oc * a.y * a.z + a.x * s,   oc * a.z * a.z + c,         0.0,
                0.0,                        0.0,                        0.0,                        1.0);
}
#endif
