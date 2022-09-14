/*
original_author: Patricio Gonzalez Vivo
description: Returns a rectangular SDF
use: rectSDF(<float2> st, <float2> size)
*/

#ifndef FNC_RECTSDF
#define FNC_RECTSDF
float rectSDF(in float2 st, in float2 s) {
    st = st * 2. - 1.;
    return max( abs(st.x / s.x),
                abs(st.y / s.y) );
}
#endif
