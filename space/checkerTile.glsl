#include "sqTile.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Return a black or white in a square checker patter
use: 
    - <vec4> checkerTile(<vec4> tile)
    - <vec4> checkerTile(<vec2> st [, <vec2> scale])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
*/

#ifndef FNC_CHECKERTILE
#define FNC_CHECKERTILE
float checkerTile(vec4 t) {
    vec2 c = mod(t.zw,2.);
    return abs(c.x-c.y);
}

float checkerTile(vec2 v) {
    return checkerTile(sqTile(v));
}

float checkerTile(vec2 v, float s) {
    return checkerTile(v * s);
}

float checkerTile(vec2 v, vec2 s) {
    return checkerTile(v * s);
}
#endif