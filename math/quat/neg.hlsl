#include "type.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion negative. 
use: <QUAT> quatNeg(<QUAT> a) 
*/

#ifndef FNC_QUATNEG
#define FNC_QUATNEG
QUAT quatNeg(QUAT q) { return QUAT(-q.xyz, -q.w); }
#endif