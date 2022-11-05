#include "levels/inputRange.glsl"
#include "levels/outputRange.glsl"
#include "levels/gamma.glsl"

/*
original_author: Johan Ismael
description: |
  Combines inputRange, outputRange and gamma functions into one
  Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levels(<vec3|vec4> color, <float|vec3> minInput, <float|vec3> gamma, <float|vec3 maxInput, <float|vec3 minOutput, <float|vec3 maxOutput)
*/

#ifndef FNC_LEVELS
#define FNC_LEVELS
vec3 levels(in vec3 color, in vec3 minInput, in vec3 gamma, in vec3 maxInput, in vec3 minOutput, in vec3 maxOutput) {
  return levelsOutputRange( levelsGamma( levelsInputRange(color, minInput, maxInput), gamma), minOutput, maxOutput);
}

vec3 levels(in vec3 color, in float minInput, in float gamma, in float maxInput, in float minOutput, in float maxOutput) {
  return levels(color, vec3(minInput), vec3(gamma), vec3(maxInput), vec3(minOutput), vec3(maxOutput));
}

vec4 levels(in vec4 color, in vec3 minInput, in vec3 gamma, in vec3 maxInput, in vec3 minOutput, in vec3 maxOutput) {
  return vec4(levels(color.rgb, minInput, gamma, maxInput, minOutput, maxOutput), color.a);
}

vec4 levels(in vec4 color, in float minInput, in float gamma, in float maxInput, in float minOutput, in float maxOutput) {
  return vec4(levels(color.rgb, minInput, gamma, maxInput, minOutput, maxOutput), color.a);
}
#endif