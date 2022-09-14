#include "rectSDF.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Returns a cross-shaped SDF
use: crossSDF(<vec2> st, size s)
*/

#ifndef FNC_CROSSSDF
#define FNC_CROSSSDF
float crossSDF(in vec2 st, in float s) {
    vec2 size = vec2(.25, s);
    return min(rectSDF(st.xy, size.xy),
               rectSDF(st.xy, size.yx));
}
#endif
