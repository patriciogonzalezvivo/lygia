/*
contributors: Johan Ismael
description: |
    Color input range adjustment similar to Levels adjustment tool in Photoshop
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsInputRange(<float3|float4> color, <float|float3> minInput, <float|float3> maxInput)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LEVELSINPUTRANGE
#define FNC_LEVELSINPUTRANGE
float3 levelsInputRange(in float3 color, in float3 minInput, in float3 maxInput) {
  return min(max(color - minInput, float3(0., 0., 0.)) / (maxInput - minInput), float3(1., 1., 1.));
}

float4 levelsInputRange(in float4 color, in float3 minInput, in float3 maxInput) {
  return float4(levelsInputRange(color.rgb, minInput, maxInput), color.a);
}

float3 levelsInputRange(in float3 color, in float minInput, in float maxInput) {
  return levelsInputRange(color, float3(minInput, minInput, minInput), float3(maxInput, maxInput, maxInput));
}

float4 levelsInputRange(in float4 color, in float minInput, in float maxInput) {
  return float4(levelsInputRange(color.rgb, minInput, maxInput), color.a);
}
#endif