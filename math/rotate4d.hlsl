/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 rotation matrix
use: <float4x4> rotate4d(<float3> axis, <float> radians)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE4D
#define FNC_ROTATE4D
float4x4 rotate4d(in float3 axis, const in float radians) {
    axis = normalize(axis);
    float s = sin(radians);
    float c = cos(radians);
    float oc = 1.0 - c;
    return float4x4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                    0.0,                                0.0,                                0.0,                                1.0);
}
#endif
