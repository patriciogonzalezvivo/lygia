#include "../math/const.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: Returns a sdf for rays with N branches
use: raysSDF(<float2> st, <int> N)
*/

#ifndef FNC_RAYSSDF
#define FNC_RAYSSDF
float raysSDF(in float2 st, in int N) {
    st -= .5;
    return frac( atan2(st.y, st.x) / TAU * float(N));
}
#endif
