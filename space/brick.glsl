/*
original_author: Patricio Gonzalez Vivo
description: brick a pattern
use: 
    - <vec2> brick(<vec2> st [, <float> scale])
    - <vec4> brick(<vec4> tiles)
*/

#ifndef FNC_BRICK
#define FNC_BRICK
vec2 brick(vec2 st) {
    st.x += step(1., mod(st.y, 2.0)) * 0.5;
    return fract(st);
}

vec2 brick(vec2 st, float scale) {
    return brick(st * scale);
}

vec4 brick(vec4 tile) {
    tile.x += mod(tile.w,2.)*.5;
    tile.z = floor(tile.z+tile.x);
    tile.x = fract(tile.x);
    return tile;
}
#endif