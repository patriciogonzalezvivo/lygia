#include "type.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: "Quaternion division \n"
use: <QUAT> quatDiv(<QUAT> a, <QUAT> b)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_QUATDIV
#define FNC_QUATDIV
QUAT quatDiv(QUAT q, float s) { return QUAT(q.xyz / s, q.w / s); }
#endif