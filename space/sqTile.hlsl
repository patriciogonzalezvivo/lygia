/*
contributors: Patricio Gonzalez Vivo
description: make some square tiles. XY provide coords inside of the tile. ZW provides tile coords
use: <float4> hexTile(<float2> st [, <float> scale])
*/

#ifndef FNC_SQTILE
#define FNC_SQTILE
float4 sqTile(float2 st) { return float4(frac(st), floor(st)); }
float4 sqTile(float2 st, float scale) { return sqTile(st * scale); }
#endif