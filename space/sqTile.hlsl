/*
contributors: Patricio Gonzalez Vivo
description: make some square tiles. XY provide coords inside of the tile. ZW provides tile coords
use: <float4> hexTile(<float2> st [, <float> scale])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SQTILE
#define FNC_SQTILE
float4 sqTile(float2 st) { return float4(frac(st), floor(st)); }
float4 sqTile(float2 st, float scale) { return sqTile(st * scale); }
#endif