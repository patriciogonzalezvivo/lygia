/*
original_author: Ivan Dianov
description: cartesian to polar transformation.
use: cart2polar(<float2> st)
*/

#ifndef FNC_CART2POLAR
#define FNC_CART2POLAR
float2 cart2polar(in float2 st) {
    return float2(atan2(st.y, st.x), length(st));
}
#endif