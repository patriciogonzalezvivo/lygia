/*
contributors: [Ivan Dianov, Kathy McGuiness]
description: cartesian to polar transformation.
use: <vec2|vec3> cart2polar(<vec2|vec3> st)
*/

#ifndef FNC_CART2POLAR
#define FNC_CART2POLAR

vec2 cart2polar(in vec2 st) {
    return vec2(atan(st.y, st.x), length(st));
}

/*
Add by jane00
Refer to https://en.wikipedia.org/wiki/Spherical_coordinate_system , there are two different definition for phi and theta, which are opposite to each other.

Before the changes by shadielhajj committed on 2024 Jul 26,  The physics convention(ISO convention) is followed, phi means "azimuthal angle", theta for "polar angle"
but now, The "mathematics convention" is followed, phi means "polar angle", theta for "azimuthal angle"

So be careful with the old code !!!
*/

// https://mathworld.wolfram.com/SphericalCoordinates.html
vec3 cart2polar( in vec3 st ) {
    float r = length(st);
    float phi = acos(st.z/r);
    float theta = atan(st.y, st.x);
    return vec3(r, phi, theta);
}

#endif
