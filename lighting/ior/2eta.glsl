/*
contributors: Patricio Gonzalez Vivo
description: index of refraction to ratio of index of refraction
use: <float|vec3|vec4> ior2eta(<float|vec3|vec4> ior)
*/

#ifndef FNC_IOR2ETA
#define FNC_IOR2ETA
float ior2eta( const in float ior ) { return 1.0/ior; }
vec3 ior2eta( const in vec3 ior ) { return 1.0/ior; }
vec4 ior2eta( const in vec4 ior ) { return vec4(1.0/ior.rgb, ior.a); }
#endif