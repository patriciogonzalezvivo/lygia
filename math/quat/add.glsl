#include "type.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: "Quaternion addition \n"
use: <QUAT> quatAdd(<QUAT> a, <QUAT> b)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_QUATADD
#define FNC_QUATADD
QUAT quatAdd(QUAT a, QUAT b) { return QUAT(a.xyz + b.xyz, a.w + b.w); }
#endif