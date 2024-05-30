#include "length.hlsl"
#include "div.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: returns a normalized quaternion
use: <QUAT> quatNorm(<QUAT> Q)
*/

#ifndef FNC_QUATNORM
#define FNC_QUATNORM
QUAT quatNorm(QUAT q) { return quatDiv(q, quatLength(q)); }
#endif