/*
original_author: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion 
use: decimate(<float|vec2|vec3|vec4> value, <float|vec2|vec3|vec4> presicion)
*/

#ifndef FNC_DECIMATION
#define FNC_DECIMATION

#define decimate(value, presicion) (floor(value * presicion)/presicion)

#endif