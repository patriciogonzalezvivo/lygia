/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
use: <float3x3> rotate3d(<float3> axis, <float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE3D
#define FNC_ROTATE3D
float3x3 rotate3d(in float3 axis, in float radians) {
    axis = normalize(axis);
    float s = sin(radians);
    float c = cos(radians);
    float oc = 1.0 - c;

    return float3x3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c );
}
#endif
