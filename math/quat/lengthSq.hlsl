#include "type.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Returns the squared length of a quaternion.
    
use: <QUAT> quatLengthSq(<QUAT> q) 
*/

#ifndef FNC_QUATLENGTHSQ
#define FNC_QUATLENGTHSQ
float quatLengthSq(QUAT q) { return dot(q.xyz, q.xyz) + q.w * q.w; }
#endif