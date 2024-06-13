/*
contributors: Patricio Gonzalez Vivo
description: Returns a rectangular SDF
use:
    - rectSDF(<vec2> st [, <vec2|float> size])
    - rectSDF(<vec2> st [, <vec2|float> size, float radius])
options:
    - CENTER_2D: vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
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
    #ifdef CENTER_2D
        st -= CENTER_2D;
        st *= 2.0;
    #else
        st = st * 2.0 - 1.0;
    #endif
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
