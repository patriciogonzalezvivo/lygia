#include "levels/inputRange.hlsl"
#include "levels/outputRange.hlsl"
#include "levels/gamma.hlsl"

/*
contributors: Johan Ismael
description: |
    Combines inputRange, outputRange and gamma functions into one
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsControl(<float3|float4> color, <float|float3> minInput, <float|float3> gamma, <float|float3 maxInput, <float|float3 minOutput, <float|float3 maxOutput)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/licenses
*/

#ifndef FNC_LEVELSCONTROL
#define FNC_LEVELSCONTROL
float3 levelsControl(in float3 color, in float3 minInput, in float3 gamma, in float3 maxInput, in float3 minOutput, in float3 maxOutput) {
  return levelsOutputRange(levelsGamma(levelsInputRange(color, minInput, maxInput), gamma), minOutput, maxOutput);
}

float3 levelsControl(in float3 color, in float minInput, in float gamma, in float maxInput, in float minOutput, in float maxOutput) {
  return levelsControl(color, float3(minInput, minInput, minInput), float3(gamma, gamma, gamma), float3(maxInput, maxInput, maxInput), float3(minOutput, minOutput, minOutput), float3(maxOutput, maxOutput, maxOutput));
}

float4 levelsControl(in float4 color, in float3 minInput, in float3 gamma, in float3 maxInput, in float3 minOutput, in float3 maxOutput) {
  return float4(levelsControl(color.rgb, minInput, gamma, maxInput, minOutput, maxOutput), color.a);
}

float4 levelsControl(in float4 color, in float minInput, in float gamma, in float maxInput, in float minOutput, in float maxOutput) {
  return float4(levelsControl(color.rgb, minInput, gamma, maxInput, minOutput, maxOutput), color.a);
}
#endif