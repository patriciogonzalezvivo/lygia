/*
contributors: Patricio Gonzalez Vivo
description: Returns a triangle-shaped sdf
use: triSDF(<vec2> st)
options:
    - CENTER_2D : vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_TRISDF
#define FNC_TRISDF
float triSDF(in vec2 st) {
#ifdef CENTER_2D
    st -= CENTER_2D;
    st *= 5.0;
#else
    st -= 0.5;
    st *= 5.0;
#endif
    return max(abs(st.x) * .866025 + st.y * .5, -st.y * 0.5);
}
#endif
