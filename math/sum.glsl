/*
original_author: Patricio Gonzalez Vivo
description: Sum elements of a vector
use: <float> sum(<vec2|vec3|vec4> value)
*/

#ifndef FNC_SUM
#define FNC_SUM

float sum( float v ) { return v; }
float sum( vec2 v ) { return v.x+v.y; }
float sum( vec3 v ) { return v.x+v.y+v.z; }
float sum( vec4 v ) { return v.x+v.y+v.z+v.w; }

#endif
