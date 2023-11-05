#include "../math/const.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a sdf for rays with N branches
use: raysSDF(<vec2> st, <int> N)
options:
    - CENTER_2D : vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_RAYSSDF
#define FNC_RAYSSDF
float raysSDF(in vec2 st, in int N) {
#ifdef CENTER_2D
    st -= CENTER_2D;
#else
    st -= 0.5;
#endif
    return fract(atan(st.y, st.x) / TAU * float(N));
}
#endif
