#include "type.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion substraction. 
use: <QUAT> quatNeg(<QUAT> a, <QUAT> b) 
*/

#ifndef FNC_QUATNEG
#define FNC_QUATNEG
QUAT quatSub(QUAT a, QUAT b) { return QUAT(a.xyz - b.xyz, a.w - b.w); }
#endif