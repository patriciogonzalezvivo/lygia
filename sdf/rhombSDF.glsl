#include "triSDF.glsl"

/*
description: Returns a rhomb-shaped sdf
use: rhombSDF(<vec2> st)
original_author: Patricio Gonzalez Vivo
*/

#ifndef FNC_RHOMBSDF
#define FNC_RHOMBSDF
float rhombSDF(in vec2 st) {
    return max(triSDF(st),
               triSDF(vec2(st.x, 1. - st.y)));
}
#endif
