/*
contributors: [Ivan Dianov, Kathy McGuiness]
description: cartesian to polar transformation.
use: <float2|float3> cart2polar(<float2|float3> st)
*/

#ifndef FNC_CART2POLAR
#define FNC_CART2POLAR

float2 cart2polar(in float2 st) {
    return float2(atan2(st.y, st.x), length(st));
}

// https://mathworld.wolfram.com/SphericalCoordinates.html
float3 cart2polar( in float3 st) {
    float r = length(st);
    float phi = acos(st.z/r);
    float theta = atan2(st.y, st.x);
    return float3(r, phi, theta);
}

#endif
