#include "sqTile.glsl"
#include "../math/mirror.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: mirror a tiles pattern
use: 
    - <vec4> mirrorTile(<vec4> tile)
    - <vec4> mirrorTile(<vec2> st [, <float> scale])
*/

#ifndef FNC_mirrorTILE
#define FNC_mirrorTILE

vec4 mirrorTile(vec4 tile) { return vec4( mirror(tile.xy), tile.z, tile.w); }
vec4 mirrorTile(vec2 st) { return mirrorTile(sqTile(st)); }
vec4 mirrorTile(vec2 st, float scale) { return mirrorTile(st * scale); }

#endif