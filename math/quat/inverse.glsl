#include "div.glsl"
#include "conj.glsl"
#include "lengthSq.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Quaternion inverse 
use: <QUAT> quatDiv(<QUAT> a, <QUAT> b) 
*/

#ifndef FNC_QUATINVERSE
#define FNC_QUATINVERSE
QUAT quatInverse(QUAT q) { return quatDiv(quatConj(q), quatLengthSq(q)); }
#endif