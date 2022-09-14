#include "../math/const.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Returns a star-shaped sdf with V branches
use: starSDF(<vec2> st, <int> V, <float> scale)
*/

#ifndef FNC_STARSDF
#define FNC_STARSDF
float starSDF(in vec2 st, in int V, in float s) {
    st = st * 4. - 2.;
    float a = atan(st.y, st.x) / TAU;
    float seg = a * float(V);
    a = ((floor(seg) + .5) / float(V) +
        mix(s, -s, step(.5, fract(seg))))
        * TAU;
    return abs(dot(vec2(cos(a), sin(a)),
                   st));
}
#endif
