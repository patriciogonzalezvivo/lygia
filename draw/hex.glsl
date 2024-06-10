
#include "../sdf/hexSDF.glsl"

#include "fill.glsl"
#include "stroke.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Draw a hexagon filled or not.
use: hex(<vec2> st, <float> size [, <float> width])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HEX
#define FNC_HEX
float hex(vec2 st, float size) {
    return fill(hexSDF(st), size);
}

float hex(vec2 st, float size, float strokeWidth) {
    return stroke(hexSDF(st), size, strokeWidth);
}
#endif