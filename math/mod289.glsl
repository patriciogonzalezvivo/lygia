/*
original_author: [Ian McEwan, Ashima Arts]
description: modulus of 289
use: mod289(<float|vec2|vec3|vec4> x)
*/

#ifndef FNC_MOD289
#define FNC_MOD289
float mod289(in float x) {
  return x - floor(x * (1. / 289.)) * 289.;
}

vec2 mod289(in vec2 x) {
  return x - floor(x * (1. / 289.)) * 289.;
}

vec3 mod289(in vec3 x) {
  return x - floor(x * (1. / 289.)) * 289.;
}

vec4 mod289(in vec4 x) {
  return x - floor(x * (1. / 289.)) * 289.;
}
#endif
