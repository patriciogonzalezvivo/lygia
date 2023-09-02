#include "type.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: given a quaternion, returns its conjugate
use: <QUAT> quatConj(<QUAT> Q)
*/

#ifndef FNC_QUATCONJ
#define FNC_QUATCONJ

QUAT quatConj(QUAT q) {
    return QUAT(-q.x, -q.y, -q.z, q.w);
}

#endif