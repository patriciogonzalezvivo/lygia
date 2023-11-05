#include "../math/const.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a sdf for a regular polygon with V sides.
use: polySDF(<vec2> st, int V)
options:
    - CENTER_2D : vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_POLYSDF
#define FNC_POLYSDF
float polySDF(in vec2 st, in int V) {
#ifdef CENTER_2D
    st -= CENTER_2D;
    st *= 2.0;
#else
    st = st * 2.0 - 1.0;
#endif
    float a = atan(st.x, st.y) + PI;
    float r = length(st);
    float v = TAU / float(V);
    return cos(floor(.5 + a / v) * v - a ) * r;
}
#endif
