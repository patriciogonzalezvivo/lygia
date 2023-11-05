/*
contributors: Patricio Gonzalez Vivo
description: make some square tiles. XY provide coords inside of the tile. ZW provides tile coords
use: <vec4> hexTile(<vec2> st [, <float> scale])
*/

#ifndef FNC_SQTILE
#define FNC_SQTILE
vec4 sqTile(vec2 st) { return vec4(fract(st), floor(st)); }
vec4 sqTile(vec2 st, float scale) { return sqTile(st * scale); }
#endif