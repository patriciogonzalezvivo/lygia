#include "triSDF.glsl"
#include "../space/scale.glsl"

/*
description: Returns a rhomb-shaped sdf
use: rhombSDF(<vec2> st)
contributors: Patricio Gonzalez Vivo
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_RHOMBSDF
#define FNC_RHOMBSDF
float rhombSDF(in vec2 st) {
    float offset = 1.0;
    #ifdef CENTER_2D
    offset = CENTER_2D.y * 2.0;
    #endif 
    return max(triSDF(st),
               triSDF(vec2(st.x, offset-st.y)));

}
#endif
