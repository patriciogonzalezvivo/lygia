#include "sqTile.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Brick a squared pattern'
use:
    - <vec2> brickTile(<vec2> st [, <float> scale])
    - <vec4> brickTile(<vec4> tiles)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_BRICKTILE
#define FNC_BRICKTILE
vec4 brickTile(vec4 t) {
    t.x += mod(t.w,2.)*.5;
    t.z = floor(t.z+t.x);
    t.x = fract(t.x);
    return t;
}

vec4 brickTile(vec2 st) {
    return brickTile(sqTile(st));
}

vec4 brickTile(vec2 st, float s) {
    return brickTile(st * s);
}

vec4 brickTile(vec2 st, vec2 s) {
    return brickTile(st * s);
}

#endif