/*
original_author: Ivan Dianov
description: cartesian to polar transformation.
use: cart2polar(<vec2> st)
*/

#ifndef FNC_CART2POLAR
#define FNC_CART2POLAR
vec2 cart2polar(in vec2 st) {
    return vec2(atan(st.y, st.x), length(st));
}
#endif