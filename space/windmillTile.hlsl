#include "../math/const.hlsl"
#include "rotate.hlsl"
#include "sqTile.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Rotate tiles (in a squared grid pattern) by 45 degrees'
use:
    - <float4> windmillTile(<float4> tiles[, <float> fullturn = TAU])
    - <float2> windmillTile(<float2> st [, <float|float2> scale])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_WINDMILLTILE
#define FNC_WINDMILLTILE
float4 windmillTile(float4 tile, float turn) {
    float a = ( abs(mod(tile.z, 2.0)-
                    mod(tile.w, 2.0))+
                mod(tile.w, 2.0) * 2.0)*
                0.25;
    return float4(rotate(tile.xy, a * turn), tile.zw);
}

float4 windmillTile(float4 tile) {
    return windmillTile(tile, TAU);
}

float4 windmillTile(float2 st) {
    return windmillTile(sqTile(st));
}

float4 windmillTile(float2 st, float scale) {
    return windmillTile(st * scale);
}

float4 windmillTile(float2 st, float2 scale) {
    return windmillTile(st * scale);
}
#endif