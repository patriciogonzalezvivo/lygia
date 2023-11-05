/*
contributors: Patricio Gonzalez Vivo
description: modify brightness and contrast
use: brightnessContrast(<float|vec3|vec4> color, <float> brightness, <float> amcontrastount)
*/

#ifndef FNC_BRIGHTNESSCONTRAST
#define FNC_BRIGHTNESSCONTRAST
float brightnessContrast( float v, float b, float c ) { return ( v - 0.5 ) * c + 0.5 + b; }
vec3 brightnessContrast( vec3 v, float b, float c ) { return ( v - 0.5 ) * c + 0.5 + b; }
vec4 brightnessContrast( vec4 v, float b, float c ) { return vec4(( v.rgb - 0.5 ) * c + 0.5 + b, v.a); }
#endif