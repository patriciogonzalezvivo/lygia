#include "../math/const.glsl"

/*
contributors: Inigo Quiles
description:  Procedural generation of color palette algorithm explained here http://www.iquilezles.org/www/articles/palettes/palettes.htm
use: <vec3|vec4> palette(<float> t, <vec3|vec4> a, <vec3|vec4> b, <vec3|vec4> c, <vec3|vec4> d)
*/

#ifndef FNC_PALETTE
#define FNC_PALETTE
vec3 palette (in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) { return a + b * cos(TAU * ( c * t + d )); }
vec4 palette (in float t, in vec4 a, in vec4 b, in vec4 c, in vec4 d) { return a + b * cos(TAU * ( c * t + d )); }
#endif
