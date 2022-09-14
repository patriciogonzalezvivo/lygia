/*
original_author: Patricio Gonzalez Vivo
description: Returns a heart shaped SDF
use: heartSDF(<vec2> st)
*/

#ifndef FNC_HEARTSDF
#define FNC_HEARTSDF
float heartSDF(vec2 st) {
    st -= vec2(0.5, 0.8);
    float r = length(st) * 5.0;
    st = normalize(st);
    return r - ((st.y * pow(abs(st.x), 0.67)) / (st.y + 1.5) - (2.0) * st.y + 1.26);
}
#endif