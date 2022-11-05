/*
original_author: Johan Ismael
description: |
  Color output range adjustment similar to Levels adjusment in Photoshop
  Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsOutputRange(<vec3|vec4> color, float minOutput, float maxOutput)
*/

#ifndef FNC_LEVELSOUTPUTRANGE
#define FNC_LEVELSOUTPUTRANGE
vec3 levelsOutputRange(in vec3 color, in vec3 minOutput, in vec3 maxOutput) {
  return mix(minOutput, maxOutput, color);
}

vec4 levelsOutputRange(in vec4 color, in vec3 minOutput, in vec3 maxOutput) {
  return vec4(levelsOutputRange(color.rgb, minOutput, maxOutput), color.a);
}

vec3 levelsOutputRange(in vec3 color, in float minOutput, in float maxOutput) {
  return levelsOutputRange(color, vec3(minOutput), vec3(maxOutput));
}

vec4 levelsOutputRange(in vec4 color, in float minOutput, in float maxOutput) {
  return vec4(levelsOutputRange(color.rgb, minOutput, maxOutput), color.a);
}
#endif