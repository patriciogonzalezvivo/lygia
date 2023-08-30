#include "sqTile.glsl"
#include "../math/mirror.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: mirror a tiles pattern
use: 
    - <vec4> mirrorTile(<vec4> tile)
    - <vec4> mirrorTile(<vec2> st [, <float|vec2> scale])
    - <vec4> mirrorXTile(<vec4> tile)
    - <vec4> mirrorXTile(<vec2> st [, <float|vec2> scale])
    - <vec4> mirrorYTile(<vec4> tile)
    - <vec4> mirrorYTile(<vec2> st [, <float|vec2> scale])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_tiles.frag
*/

#ifndef FNC_MIRRORTILE
#define FNC_MIRRORTILE

vec4 mirrorTile(vec4 tile) { return vec4( mirror(tile.xy + tile.zw), tile.z, tile.w); }
vec4 mirrorTile(vec2 st) { return mirrorTile(sqTile(st)); }
vec4 mirrorTile(vec2 st, float scale) { return mirrorTile(st * scale); }
vec4 mirrorTile(vec2 st, vec2 scale) { return mirrorTile(st * scale); }

vec4 mirrorXTile(vec4 tile) { return vec4( mirror(tile.x + tile.z), tile.y, tile.z, tile.w); }
vec4 mirrorXTile(vec2 st) { return mirrorXTile(sqTile(st)); }
vec4 mirrorXTile(vec2 st, float scale) { return mirrorXTile(st * scale); }
vec4 mirrorXTile(vec2 st, vec2 scale) { return mirrorXTile(st * scale); }

vec4 mirrorYTile(vec4 tile) { return vec4( tile.x, mirror(tile.y + tile.w), tile.z, tile.w); }
vec4 mirrorYTile(vec2 st) { return mirrorYTile(sqTile(st)); }
vec4 mirrorYTile(vec2 st, float scale) { return mirrorYTile(st * scale); }
vec4 mirrorYTile(vec2 st, vec2 scale) { return mirrorYTile(st * scale); }

#endif