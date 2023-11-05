/*
contributors:  Inigo Quiles
description: generate the SDF of a sphere
use: <float> sphereSDF( in <vec3> pos[], in <float> size] ) 
*/

#ifndef FNC_SPHERESDF
#define FNC_SPHERESDF
float sphereSDF(vec3 p) { return length(p); }
float sphereSDF(vec3 p, float s) { return sphereSDF(p) - s; }
#endif