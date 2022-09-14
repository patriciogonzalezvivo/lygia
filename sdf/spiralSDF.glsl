/*
original_author: Patricio Gonzalez Vivo
description: Returns a spiral SDF
use: spiralSDF(<vec2> st, <float> turns)
*/

#ifndef FNC_SPIRALSDF
#define FNC_SPIRALSDF
float spiralSDF(vec2 st, float t) {
    st -= 0.5;
    float r = dot(st, st);
    float a = atan(st.y, st.x);
    return abs(sin(fract(log(r) * t + a * 0.159)));
}
#endif