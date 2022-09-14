/*
original_author: Patricio Gonzalez Vivo
description: Returns a rectangular SDF
use: rectSDF(<vec2> st, <vec2> size)
*/

#ifndef FNC_RECTSDF
#define FNC_RECTSDF

float rectSDF(vec2 p, vec2 b, float r) {
    vec2 d = abs(p - 0.5) * 4.2 - b + vec2(r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - r;   
}

float rectSDF(vec2 p, float b, float r) {
    return rectSDF(p, vec2(b), r);
}

float rectSDF(in vec2 st, in vec2 s) {
    st = st * 2. - 1.;
    return max( abs(st.x / s.x),
                abs(st.y / s.y) );
}

float rectSDF(in vec2 st, in float s) {
    return rectSDF(st, vec2(s) );
}

float rectSDF(in vec2 st) {
    return rectSDF(st, vec2(1.0));
}

#endif
