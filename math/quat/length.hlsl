#include "lengthSq.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Returns the lenght of a quaternion
    
use: <QUAT> quatLength(<QUAT> q) 
*/

#ifndef FNC_QUADLENGTH
#define FNC_QUADLENGTH
float quatLength(QUAT q) { return sqrt(quatLengthSq(q)); }
#endif