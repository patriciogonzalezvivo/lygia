/*
original_author: Patricio Gonzalez Vivo
description: Returns a circle-shaped SDF.
use: circleSDF(float2 st[, float2 center])
options:
    CIRCLESDF_LENGHT_FNC(POS_UV) : function used to calculate the SDF, defaults to GLSL length function, use lengthSq for a different slope
*/

#ifndef CIRCLESDF_LENGHT_FNC
#define CIRCLESDF_LENGHT_FNC(POS_UV) length(POS_UV)
#endif

#ifndef FNC_CIRCLESDF
#define FNC_CIRCLESDF

float circleSDF(in float2 st, in float2 center) {
    return CIRCLESDF_LENGHT_FNC(st - center) * 2.;
}

float circleSDF(in float2 st) {
    return circleSDF(st, float2(0.5, 0.5));
}

#endif
