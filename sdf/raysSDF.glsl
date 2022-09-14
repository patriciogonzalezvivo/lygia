#include "../math/const.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Returns a sdf for rays with N branches
use: raysSDF(<vec2> st, <int> N)
*/

#ifndef FNC_RAYSSDF
#define FNC_RAYSSDF
float raysSDF(in vec2 st, in int N) {
    st -= .5;
    return fract(atan(st.y, st.x) / TAU * float(N));
}
#endif
