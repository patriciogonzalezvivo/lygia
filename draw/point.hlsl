
#include "digits.hlsl"
#include "circle.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Draws all the digits of a matrix, useful for debugging.
use:
    - <float4> point(<float2> st, <float2> pos)
    - <float4> point(<float2> st, <float4x4> P, <float4x4> V, <float3> pos)
options:
    DIGITS_DECIMALS: number of decimals after the point, defaults to 2
    DIGITS_SIZE: size of the font, defaults to float2(.025, .025)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_DRAW_POINT
#define FNC_DRAW_POINT

float4 point(in float2 st, float2 pos, float3 color, float radius) {
    float4 rta = float4(0.0, 0.0, 0.0, 0.0);
    float2 st_p = st - pos;

    rta += float4(color, 1.0) * circle(st_p + 0.5, radius);
    st_p -= DIGITS_SIZE * float2(0.0, 0.5);
    float2 size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * float2(2.0, 0.5);
    rta.a += 0.5 * step(0.0, st_p.x) * step(st_p.x, size.x) * step(-DIGITS_SIZE.y * 0.5, st_p.y) * step(st_p.y, size.y);
    rta += digits(st_p, pos);

    return rta;
}

float4 point(in float2 st, float4x4 P, float4x4 V, float3 pos, float3 color, float radius) {
    float4 rta = float4(0.0, 0.0, 0.0, 0.0);
    float4 pos4 = mult(P, mult(V * float4(pos, 1.0)));
    float2 p = pos4.xy / pos4.w;

    #ifdef DEBUG_FLIPPED_SPACE
    float2 st_p = st + (p.xy * 0.5 - 0.5);
    #else
    float2 st_p = st - (p.xy * 0.5 + 0.5);
    #endif

    rta += float4(color, 1.0) * circle(st_p + 0.5, radius);
    st_p -= DIGITS_SIZE * float2(0.0, 0.5);
    float2 size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * float2(3.0, 0.5);
    rta.a += 0.5 * step(0.0, st_p.x) * step(st_p.x, size.x) * step(-DIGITS_SIZE.y * 0.5, st_p.y) * step(st_p.y, size.y);
    rta += digits(st_p, pos);
    
    return rta;
}

float4 point(in float2 st, float2 pos) { return point(st, pos, float3(1.0, 0.0, 0.0), 0.02); }
float4 point(in float2 st, float4x4 P, float4x4 V, float3 pos) { return point(st, P, V, pos, float3(1.0, 0.0, 0.0), 0.02); }

#endif