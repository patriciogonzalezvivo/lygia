#include "rectSDF.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a cross-shaped SDF
use: crossSDF(<vec2> st, size s)
options:
    - CENTER_2D : vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_CROSSSDF
#define FNC_CROSSSDF
float crossSDF(in vec2 st, in float s) {
    vec2 size = vec2(.25, s);
    return min(rectSDF(st.xy, size.xy),
               rectSDF(st.xy, size.yx));
}
#endif
