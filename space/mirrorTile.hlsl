#include "sqTile.hlsl"
#include "../math/mirror.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: mirror a tiles pattern
use:
    - <float4> mirrorTile(<float4> tile)
    - <float4> mirrorTile(<float2> st [, <float|float2> scale])
    - <float4> mirrorXTile(<float4> tile)
    - <float4> mirrorXTile(<float2> st [, <float|float2> scale])
    - <float4> mirrorYTile(<float4> tile)
    - <float4> mirrorYTile(<float2> st [, <float|float2> scale])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MIRRORTILE
#define FNC_MIRRORTILE
float4 mirrorTile(float4 tile) { return float4( mirror(tile.xy + tile.zw), tile.z, tile.w); }
float4 mirrorTile(float2 st) { return mirrorTile(sqTile(st)); }
float4 mirrorTile(float2 st, float scale) { return mirrorTile(st * scale); }
float4 mirrorTile(float2 st, float2 scale) { return mirrorTile(st * scale); }

float4 mirrorXTile(float4 tile) { return float4( mirror(tile.x + tile.z), tile.y, tile.z, tile.w); }
float4 mirrorXTile(float2 st) { return mirrorXTile(sqTile(st)); }
float4 mirrorXTile(float2 st, float scale) { return mirrorXTile(st * scale); }
float4 mirrorXTile(float2 st, float2 scale) { return mirrorXTile(st * scale); }

float4 mirrorYTile(float4 tile) { return float4( tile.x, mirror(tile.y + tile.w), tile.z, tile.w); }
float4 mirrorYTile(float2 st) { return mirrorYTile(sqTile(st)); }
float4 mirrorYTile(float2 st, float scale) { return mirrorYTile(st * scale); }
float4 mirrorYTile(float2 st, float2 scale) { return mirrorYTile(st * scale); }

#endif