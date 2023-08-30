/*
original_author: Patricio Gonzalez Vivo
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
float checkerTile(vec4 tile) {
    vec2 c = mod(tile.zw,2.);
    return abs(c.x-c.y);
}

float checkerTile(vec2 st) {
    return checkerTile(sqTile(st));
}

float checkerTile(vec2 st, float scale) {
    return checkerTile(st * scale);
}

float checkerTile(vec2 st, vec2 scale) {
    return checkerTile(st * scale);
}
#endif