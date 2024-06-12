#include "type.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Returns the squared length of a quaternion.'
use: <QUAT> quatLengthSq(<QUAT> q)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_QUATLENGTHSQ
#define FNC_QUATLENGTHSQ
float quatLengthSq(QUAT q) { return dot(q.xyz, q.xyz) + q.w * q.w; }
#endif