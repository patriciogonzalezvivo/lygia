/*
original_author: Patricio Gonzalez Vivo
description: Returns a triangle-shaped sdf
use: triSDF(<float2> st)
*/

#ifndef FNC_TRISDF
#define FNC_TRISDF
float triSDF(in float2 st) {
    st = (st * 2. - 1.) * 2.;
    return max(abs(st.x) * .866025 + st.y * .5, -st.y * .5);
}
#endif
