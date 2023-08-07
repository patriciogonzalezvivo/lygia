#include "../math/const.glsl"

/*
description: |
    It creates a kaleisdescope pattern
    Adapted from [Koch Snowflake tutorial](https://www.shadertoy.com/view/tdcGDj)
use: 
    - <vec2> kaleidescope(<vec2> st)
*/

#ifndef FNC_KALEIDESCOPE
#define FNC_KALEIDESCOPE
vec2 kaleidescope( vec2 st ) {
    st = st * 2.0 - 1.0;
    st = abs(st); 
    st.y += -0.288;
    vec2 n = vec2(-0.866, 0.5);
    float d = dot(st- 0.5, n);
    st -= n * max(0.0, d) * 2.0;
    st.y -= -0.433; 
    n = vec2(-0.5, 0.866);
    d = dot(st, n);
    st -= n * min(0.0, d) * 2.0;
    st.y -= 0.288; 
    return st;
}
#endif