#include "../math/const.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a sdf for rays with N branches
use: raysSDF(<float2> st, <int> N)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RAYSSDF
#define FNC_RAYSSDF
float raysSDF(in float2 st, in int N) {
#ifdef CENTER_2D
    st -= CENTER_2D;
#else
    st -= 0.5;
#endif
    return frac(atan2(st.y, st.x) / TAU * float(N));
}
#endif
