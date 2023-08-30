#include "../math/const.glsl"
#include "rotate.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Rotate tiles (in a squared grid pattern) by 45 degrees
use: 
    - <vec4> windmillTile(<vec4> tiles[, <float> fullturn = TAU])
    - <vec2> windmillTile(<vec2> st [, <float|vec2> scale])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
*/

#ifndef FNC_WINDMILLTILE
#define FNC_WINDMILLTILE
vec4 windmillTile(vec4 tile, float turn) {
    float a = ( abs(mod(tile.z, 2.0)-
                    mod(tile.w, 2.0))+
                mod(tile.w, 2.0) * 2.0)*
                0.25;
    return vec4(rotate(tile.xy, a * turn), tile.zw);
}

vec4 windmillTile(vec4 tile) {
    return windmillTile(tile, TAU);
}

vec4 windmillTile(vec2 st) {
    return windmillTile(sqTile(st));
}

vec4 windmillTile(vec2 st, float scale) {
    return windmillTile(st * scale);
}

vec4 windmillTile(vec2 st, vec2 scale) {
    return windmillTile(st * scale);
}
#endif