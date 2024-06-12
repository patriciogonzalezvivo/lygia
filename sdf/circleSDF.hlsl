/*
contributors: Patricio Gonzalez Vivo
description: Returns a circle-shaped SDF.
use: circleSDF(float2 st[, float2 center])
options:
    - CIRCLESDF_LENGHT_FNC(POS_UV): function used to calculate the SDF, defaults to GLSL length function, use lengthSq for a different slope
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef CIRCLESDF_FNC
#define CIRCLESDF_FNC(POS_UV) length(POS_UV)
#endif

#ifndef FNC_CIRCLESDF
#define FNC_CIRCLESDF

float circleSDF(in float2 st) {
#ifdef CENTER_2D
    st -= CENTER_2D;
#else
    st -= 0.5;
#endif
    return CIRCLESDF_FNC(st) * 2.0;
}

#endif
