#include "../math/const.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: Returns a sdf for a regular polygon with V sides.
use: polySDF(<float2> st, int V)
*/

#ifndef FNC_POLYSDF
#define FNC_POLYSDF
float polySDF(in float2 st, in int V) {
    st = st * 2. - 1.;
    float a = atan2(st.x, st.y) + PI;
    float r = length(st);
    float v = TAU / float(V);
    return cos(floor(.5 + a / v) * v - a ) * r;
}
#endif
