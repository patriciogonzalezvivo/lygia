#include "sqTile.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Return a black or white in a square checker patter'
use:
    - <float4> checkerTile(<float4> tile)
    - <float4> checkerTile(<float2> st [, <float2> scale])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_CHECKERTILE
#define FNC_CHECKERTILE
float checkerTile(float4 tile) {
    float2 c = mod(tile.zw, 2.0);
    return abs(c.x-c.y);
}

float checkerTile(float2 st) {
    return checkerTile(sqTile(st));
}

float checkerTile(float2 st, float scale) {
    return checkerTile(st * scale);
}

float checkerTile(float2 st, float2 scale) {
    return checkerTile(st * scale);
}
#endif