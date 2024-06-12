/*
contributors: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: <float|float2|float3|float4> mix(<float|float2|float3|float4> a, <float|float2|float3|float4> b, <float|float2|float3|float4> c [, <float|float2|float3|float4> d], <float> pct)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MIX
#define FNC_MIX

// Overloads for mixing two values
float mix(float a, float b, float pct) {
    return lerp(a, b, pct);
}

float2 mix(float2 a, float2 b, float pct) {
    return lerp(a, b, pct);
}

float3 mix(float3 a, float3 b, float pct) {
    return lerp(a, b, pct);
}

float4 mix(float4 a, float4 b, float pct) {
    return lerp(a, b, pct);
}

// Overloads for mixing three values
float mix(float a, float b, float c, float pct) {
    return lerp(lerp(a, b, 2. * pct), lerp(b, c, 2. * (max(pct, .5) - .5)), step(.5, pct));
}

float2 mix(float2 a, float2 b, float2 c, float pct) {
    return lerp(lerp(a, b, 2. * pct), lerp(b, c, 2. * (max(pct, .5) - .5)), step(.5, pct));
}

float3 mix(float3 a, float3 b, float3 c, float pct) {
    return lerp(lerp(a, b, 2. * pct), lerp(b, c, 2. * (max(pct, .5) - .5)), step(.5, pct));
}

float4 mix(float4 a, float4 b, float4 c, float pct) {
    return lerp(lerp(a, b, 2. * pct), lerp(b, c, 2. * (max(pct, .5) - .5)), step(.5, pct));
}

// Overloads for mixing four values
float mix(float a, float b, float c, float d, float pct) {
    return lerp(lerp(a, b, 3. * pct), lerp(b, lerp(c, d, 3. * (max(pct, .66) - .66)), 3. * (clamp(pct, .33, .66) - .33)), step(.33, pct));
}

float2 mix(float2 a, float2 b, float2 c, float2 d, float pct) {
    return lerp(lerp(a, b, 3. * pct), lerp(b, lerp(c, d, 3. * (max(pct, .66) - .66)), 3. * (clamp(pct, .33, .66) - .33)), step(.33, pct));
}

float3 mix(float3 a, float3 b, float3 c, float3 d, float pct) {
    return lerp(lerp(a, b, 3. * pct), lerp(b, lerp(c, d, 3. * (max(pct, .66) - .66)), 3. * (clamp(pct, .33, .66) - .33)), step(.33, pct));
}

float4 mix(float4 a, float4 b, float4 c, float4 d, float pct) {
    return lerp(lerp(a, b, 3. * pct), lerp(b, lerp(c, d, 3. * (max(pct, .66) - .66)), 3. * (clamp(pct, .33, .66) - .33)), step(.33, pct));
}

#endif
