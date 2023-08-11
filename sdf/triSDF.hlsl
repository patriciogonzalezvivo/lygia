/*
original_author: Patricio Gonzalez Vivo
description: Returns a triangle-shaped sdf
use: triSDF(<float2> st)
*/

#ifndef FNC_TRISDF
#define FNC_TRISDF
float triSDF(in float2 st, float2 center) {
    st -= center;
    st = st*2.0 ;
    return max(abs(st.x) * .866025 + st.y * .5, -st.y * .5) - 0.25;
}

float triSDF(in float2 st) {
#ifdef CENTER_2D
    return triSDF(st, CENTER_2D);
#else
    return triSDF(st, float2(0.5, 0.5));
#endif
}
#endif
