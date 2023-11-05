#include "../math/const.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a sdf for a regular polygon with V sides.
use: polySDF(<float2> st, int V)
*/

#ifndef FNC_POLYSDF
#define FNC_POLYSDF
float polySDF(in float2 st, in int V) {
#ifdef CENTER_2D
    st -= CENTER_2D;
    st *= 2.0;
#else
    st = st * 2.0 - 1.0;
#endif
    float a = atan2(st.x, st.y) + PI;
    float r = length(st);
    float v = TAU / float(V);
    return cos(floor(.5 + a / v) * v - a ) * r;
}
#endif
