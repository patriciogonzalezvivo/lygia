/*
original_author: Patricio Gonzalez Vivo
description: |
    Brick a squared pattern
use: 
    - <vec2> brickTile(<vec2> st [, <float> scale])
    - <vec4> brickTile(<vec4> tiles)
*/

#ifndef FNC_BRICKTILE
#define FNC_BRICKTILE
vec4 brickTile(vec4 tile) {
    tile.x += mod(tile.w,2.)*.5;
    tile.z = floor(tile.z+tile.x);
    tile.x = fract(tile.x);
    return tile;
}

vec4 brickTile(vec2 st) {
    return brickTile(sqTile(st));
}

vec4 brickTile(vec2 st, float scale) {
    return brickTile(st * scale);
}

vec4 brickTile(vec2 st, vec2 scale) {
    return brickTile(st * scale);
}

#endif