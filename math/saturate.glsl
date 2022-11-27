/*
original_author: Patricio Gonzalez Vivo
description: clamp a value between 0 and 1
use: saturation(<float|vec2|vec3|vec4> value)
*/

#ifndef FNC_SATURATE
#define FNC_SATURATE
// #define saturate(x) clamp(x, 0.0, 1.0)
float saturate( const in float x){ return clamp(x, 0.0, 1.0); }
vec2  saturate( const in vec2 x ){ return clamp(x, 0.0, 1.0); }
vec3  saturate( const in vec3 x ){ return clamp(x, 0.0, 1.0); }
vec4  saturate( const in vec4 x ){ return clamp(x, 0.0, 1.0); }
#endif