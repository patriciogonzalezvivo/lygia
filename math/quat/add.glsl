#include "type.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion addition 
use: <QUAT> quatAdd(<QUAT> a, <QUAT> b) 
*/

#ifndef FNC_QUATADD
#define FNC_QUATADD
QUAT quatAdd(QUAT a, QUAT b) { return QUAT(a.xyz + b.xyz, a.w + b.w); }
#endif