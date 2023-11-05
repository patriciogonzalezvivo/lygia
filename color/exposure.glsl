/*
contributors: Patricio Gonzalez Vivo
description: change the exposure of a color
use: exposure(<float|vec3|vec4> color, float amount)
*/

#ifndef FNC_EXPOSURE
#define FNC_EXPOSURE
float exposure(float v, float a) { return v * pow(2., a); }
vec3 exposure(vec3 v, float a) { return v * pow(2., a); }
vec4 exposure(vec4 v, float a) { return vec4(v.rgb * pow(2., a), v.a); }
#endif