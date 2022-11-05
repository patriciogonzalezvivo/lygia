/*
original_author: Johan Ismael
description: |
  Color input range adjustment similar to Levels adjusment tool in Photoshop
  Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsInputRange(<vec3|vec4> color, <float|vec3> minInput, <float|vec3> maxInput)
*/

#ifndef FNC_LEVELSINPUTRANGE
#define FNC_LEVELSINPUTRANGE
vec3 levelsInputRange(in vec3 color, in vec3 minInput, in vec3 maxInput) {
  return min(max(color - minInput, vec3(0.)) / (maxInput - minInput), vec3(1.));
}

vec4 levelsInputRange(in vec4 color, in vec3 minInput, in vec3 maxInput) {
  return vec4(levelsInputRange(color.rgb, minInput, maxInput), color.a);
}

vec3 levelsInputRange(in vec3 color, in float minInput, in float maxInput) {
  return levelsInputRange(color, vec3(minInput), vec3(maxInput));
}

vec4 levelsInputRange(in vec4 color, in float minInput, in float maxInput) {
  return vec4(levelsInputRange(color.rgb, minInput, maxInput), color.a);
}
#endif