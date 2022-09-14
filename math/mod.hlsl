/*
original_author: Patricio Gonzalez Vivo
description: An implementation of mod that matches the GLSL mod.
  Note that HLSL's fmod is different.
use: mod(<float|float2|float3|float4> value, <float|float2|float3|float4> modulus)
*/

#ifndef FNC_MOD
#define FNC_MOD
float mod(float x, float y) {
    return x - y * floor(x / y);
}

float2 mod(float2 x, float2 y) {
    return x - y * floor(x / y);
}

float3 mod(float3 x, float3 y) {
    return x - y * floor(x / y);
}

float4 mod(float4 x, float4 y) {
    return x - y * floor(x / y);
}
#endif