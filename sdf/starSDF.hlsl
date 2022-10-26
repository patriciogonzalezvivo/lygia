#include "../math/const.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: Returns a star-shaped sdf with V branches
use: starSDF(<float2> st, <int> V, <float> scale)
*/

#ifndef FNC_STARSDF
#define FNC_STARSDF
float starSDF(in float2 st, in int V, in float s) {
    st = st * 4. - 2.;
    float a = atan2(st.y, st.x) / TAU;
    float seg = a * float(V);
    a = ((floor(seg) + .5) / float(V) +
        lerp(s, -s, step(.5, frac(seg))))
        * TAU;
    return abs(dot(float2(cos(a), sin(a)),
                   st));
}
#endif
