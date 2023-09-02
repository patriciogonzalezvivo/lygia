#include "type.glsl"

#ifndef FNC_QUATCONJ
#define FNC_QUATCONJ

QUAT quatConj(QUAT q) {
    return QUAT(-q.x, -q.y, -q.z, q.w);
}

#endif