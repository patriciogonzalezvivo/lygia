/*
original_author: [Ivan Dianov, Kathy McGuiness]
description: cartesian to polar transformation.
use: <vec2|vec3> cart2polar(<vec2|vec3> st)
*/

#ifndef FNC_CART2POLAR
#define FNC_CART2POLAR

vec2 cart2polar(in vec2 st) {
    return vec2(atan(st.y, st.x), length(st));
}

vec3 cart2polar( in vec3 st ) {
   float r = length(st);
   float theta = atan(length(st.xy), st.z);
   float phi = atan(st.y, st.x);
   return vec3(r, theta, phi);
}

#endif