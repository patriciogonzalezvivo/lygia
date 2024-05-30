#include "type.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion division 
use: <QUAT> quatDiv(<QUAT> a, <QUAT> b) 
*/

#ifndef FNC_QUATDIV
#define FNC_QUATDIV
QUAT quatDiv(QUAT q, float s) { return QUAT(q.xyz / s, q.w / s); }
#endif