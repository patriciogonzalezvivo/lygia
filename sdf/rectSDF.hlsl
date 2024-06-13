/*
contributors: Patricio Gonzalez Vivo
description: Returns a rectangular SDF
use: rectSDF(<float2> st, <float2> size)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RECTSDF
#define FNC_RECTSDF
float rectSDF(float2 p, float2 b, float r) {
    float2 d = abs(p - 0.5) * 4.2 - b + float2(r, r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - r;   
}

float rectSDF(float2 p, float b, float r) {
    return rectSDF(p, float2(b, b), r);
}

float rectSDF(in float2 st, in float2 s) {
    #ifdef CENTER_2D
        st -= CENTER_2D;
        st *= 2.0;
    #else
        st = st * 2.0 - 1.0;
    #endif
    return max( abs(st.x / s.x),
                abs(st.y / s.y) );
}

float rectSDF(in float2 st, in float s) {
    return rectSDF(st, float2(s, s) );
}

float rectSDF(in float2 st) {
    return rectSDF(st, float2(1.0, 1.0));
}
#endif
