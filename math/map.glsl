#include "saturate.glsl"

/*
original_author: Johan Ismael
description: Map a value between one range to another.
use: map(<float|vec2|vec3|vec4> value, <float|vec2|vec3|vec4> inMin, <float|vec2|vec3|vec4> inMax, <float|vec2|vec3|vec4> outMin, <float|vec2|vec3|vec4> outMax)
*/

#ifndef FNC_MAP
#define FNC_MAP

float map( float value, float inMin, float inMax ) {
    return saturate( (value-inMin)/(inMax-inMin));
}

vec2 map( vec2 value, vec2 inMin, vec2 inMax ) {
    return saturate( (value-inMin)/(inMax-inMin));
}

vec3 map( vec3 value, vec3 inMin, vec3 inMax ) {
    return saturate( (value-inMin)/(inMax-inMin));
}

vec4 map( vec4 value, vec4 inMin, vec4 inMax ) {
    return saturate( (value-inMin)/(inMax-inMin));
}

float map(in float value, in float inMin, in float inMax, in float outMin, in float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec2 map(in vec2 value, in vec2 inMin, in vec2 inMax, in vec2 outMin, in vec2 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec3 map(in vec3 value, in vec3 inMin, in vec3 inMax, in vec3 outMin, in vec3 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec4 map(in vec4 value, in vec4 inMin, in vec4 inMax, in vec4 outMin, in vec4 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

#endif
