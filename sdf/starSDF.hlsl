#include "../math/const.hlsl"
#include "../space/scale.hlsl"
/*
contributors: Patricio Gonzalez Vivo
description: Returns a star-shaped sdf with V branches
use: starSDF(<float2> st, <int> V, <float> scale)
*/

#ifndef FNC_STARSDF
#define FNC_STARSDF
float starSDF(in float2 st, in int V, in float s) {
#ifdef CENTER_2D
    st -= CENTER_2D;
#else
    st -= 0.5;
#endif
    st *= 2.0;
    float a = atan2(st.y, st.x) / TAU;
    float seg = a * float(V);
    a = ((floor(seg) + 0.5) / float(V) +
        lerp(s, -s, step(0.5, frac(seg))))
        * TAU;
    return abs(dot(float2(cos(a), sin(a)),
                   st));
}

float starSDF(in float2 st, in int V) {
    return starSDF( scale(st, 12.0/float(V)), V, 0.1);
}
#endif
