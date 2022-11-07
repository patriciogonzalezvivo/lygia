#include "sqTile.glsl"
#include "../math/mirror.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: mirror a tiles pattern
use: 
    - <float4> mirrorTile(<float4> tile)
    - <float4> mirrorTile(<float2> st [, <float> scale])
*/

#ifndef FNC_mirrorTILE
#define FNC_mirrorTILE
float4 mirrorTile(float4 tile) { return float4( mirror(tile.xy), tile.z, tile.w); }
float4 mirrorTile(float2 st) { return mirrorTile(sqTile(st)); }
float4 mirrorTile(float2 st, float scale) { return mirrorTile(st * scale); }
#endif