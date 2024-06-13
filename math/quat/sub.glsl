#include "type.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: "Quaternion substraction. \n"
use: <QUAT> quatNeg(<QUAT> a, <QUAT> b)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_QUATNEG
#define FNC_QUATNEG
QUAT quatSub(QUAT a, QUAT b) { return QUAT(a.xyz - b.xyz, a.w - b.w); }
#endif