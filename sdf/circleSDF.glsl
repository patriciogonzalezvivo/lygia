/*
author: Patricio Gonzalez Vivo
description: Returns a circle-shaped SDF.
use: circleSDF(vec2 st[, vec2 center])
options:
    CIRCLESDF_FNC(POS_UV) : function used to calculate the SDF, defaults to GLSL length function, use lengthSq for a different slope
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
    Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
    https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef CIRCLESDF_FNC
#define CIRCLESDF_FNC(POS_UV) length(POS_UV)
#endif

#ifndef FNC_CIRCLESDF
#define FNC_CIRCLESDF

float circleSDF(in vec2 st, in vec2 center) {
    return CIRCLESDF_FNC(st - center) * 2.;
}

float circleSDF(in vec2 st) {
    return circleSDF(st, vec2(.5));
}

#endif
