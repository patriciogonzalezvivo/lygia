#include "../math/const.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a sdf for rays with N branches
use: raysSDF(<float2> st, <int> N)
*/

#ifndef FNC_RAYSSDF
#define FNC_RAYSSDF
float raysSDF(in vec2 st, in int N) {
#ifdef CENTER_2D
    st -= CENTER_2D;
#else
    st -= 0.5;
#endif
    return frac(atan2(st.y, st.x) / TAU * float(N));
}
#endif
