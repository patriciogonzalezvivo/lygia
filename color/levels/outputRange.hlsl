/*
contributors: Johan Ismael
description: |
  Color output range adjustment similar to Levels adjusment in Photoshop
  Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsOutputRange(<float3|float4> color, float minOutput, float maxOutput)
*/

#ifndef FNC_LEVELSOUTPUTRANGE
#define FNC_LEVELSOUTPUTRANGE
float3 levelsOutputRange(in float3 color, in float3 minOutput, in float3 maxOutput) {
  return lerp(minOutput, maxOutput, color);
}

float4 levelsOutputRange(in float4 color, in float3 minOutput, in float3 maxOutput) {
  return float4(levelsOutputRange(color.rgb, minOutput, maxOutput), color.a);
}

float3 levelsOutputRange(in float3 color, in float minOutput, in float maxOutput) {
  return levelsOutputRange(color, float3(minOutput, minOutput, minOutput), float3(maxOutput, maxOutput, maxOutput));
}

float4 levelsOutputRange(in float4 color, in float minOutput, in float maxOutput) {
  return float4(levelsOutputRange(color.rgb, minOutput, maxOutput), color.a);
}
#endif