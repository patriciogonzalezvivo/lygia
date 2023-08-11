/*
original_author: Patricio Gonzalez Vivo
description: Returns a triangle-shaped sdf
use: triSDF(<vec2> st)
options:
    - CENTER_2D : vec2, defaults to vec2(.5)
*/

#ifndef FNC_TRISDF
#define FNC_TRISDF
float triSDF(in vec2 st, vec2 center) {
    st -= center;
    st = st * 2.0 ;
    return max(abs(st.x) * .866025 + st.y * .5, -st.y * .5) - 0.25;
}

float triSDF(in vec2 st) {
#ifdef CENTER_2D
    return triSDF(st, CENTER_2D);
#else
    return triSDF(st, vec2(0.5));
#endif
}
#endif
