/*
contributors: Patricio Gonzalez Vivo
description: Returns a circle-shaped SDF.
use: circleSDF(vec2 st[, vec2 center])
options:
    - CENTER_2D : vec2, defaults to vec2(.5)
    - CIRCLESDF_FNC(POS_UV) : function used to calculate the SDF, defaults to GLSL length function, use lengthSq for a different slope
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef CIRCLESDF_FNC
#define CIRCLESDF_FNC(POS_UV) length(POS_UV)
#endif

#ifndef FNC_CIRCLESDF
#define FNC_CIRCLESDF

float circleSDF(in vec2 v) {
#ifdef CENTER_2D
    v -= CENTER_2D;
#else
    v -= 0.5;
#endif
    return CIRCLESDF_FNC(v) * 2.0;
}
#endif
