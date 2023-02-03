/*
original_author: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion 
use: decimate(<float|vec2|vec3|vec4> value, <float|vec2|vec3|vec4> presicion)
*/

#ifndef FNC_DECIMATION
#define FNC_DECIMATION

float decimate(float value, float presicion) { return floor(value * presicion)/presicion; }
vec2 decimate(vec2 value, float presicion) { return floor(value * presicion)/presicion; }
vec3 decimate(vec3 value, float presicion) { return floor(value * presicion)/presicion; }
vec4 decimate(vec4 value, float presicion) { return floor(value * presicion)/presicion; }
vec2 decimate(vec2 value, vec2 presicion) { return floor(value * presicion)/presicion; }
vec3 decimate(vec3 value, vec3 presicion) { return floor(value * presicion)/presicion; }
vec4 decimate(vec4 value, vec4 presicion) { return floor(value * presicion)/presicion; }

#endif