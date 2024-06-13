#include "type.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Quaternion multiplication. Based on http://mathworld.wolfram.com/Quaternion.html'
use: <QUAT> quatMul(<QUAT> a, <QUAT> b)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_QUATMUL
#define FNC_QUATMUL
QUAT quatMul(QUAT q1, QUAT q2) {
    return QUAT(
        q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
        q1.w * q2.w - dot(q1.xyz, q2.xyz)
    );
}

QUAT quatMul(QUAT q, float s) { return QUAT(q.xyz * s, q.w * s); }
#endif