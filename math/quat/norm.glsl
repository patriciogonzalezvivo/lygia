#include "length.glsl"
#include "div.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: returns a normalized quaternion
use: <QUAT> quatNorm(<QUAT> Q)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_QUATNORM
#define FNC_QUATNORM
QUAT quatNorm(QUAT q) { return quatDiv(q, quatLength(q)); }
#endif