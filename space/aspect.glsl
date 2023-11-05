/*
contributors: Patricio Gonzalez Vivo
description: |
    Fix the aspect ratio of a space keeping things squared for you.
use: <vec2> aspect(<vec2> st, <vec2> st_size)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_ASPECT
#define FNC_ASPECT
vec2 aspect(vec2 st, vec2 s) {
    st.x = st.x * (s.x / s.y);
    return st;
}
#endif