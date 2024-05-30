/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <mat3> rotate3d(<vec3> axis, <float> radians)
*/

#ifndef FNC_ROTATE3D
#define FNC_ROTATE3D
mat3 rotate3d(in vec3 a, const in float r) {
    a = normalize(a);
    float s = sin(r);
    float c = cos(r);
    float oc = 1.0 - c;
    return mat3(oc * a.x * a.x + c,           oc * a.x * a.y - a.z * s,  oc * a.z * a.x + a.y * s,
                oc * a.x * a.y + a.z * s,  oc * a.y * a.y + c,           oc * a.y * a.z - a.x * s,
                oc * a.z * a.x - a.y * s,  oc * a.y * a.z + a.x * s,  oc * a.z * a.z + c );
}
#endif
