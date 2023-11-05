/*
contributors:  Inigo Quiles
description: generate the SDF of a sphere
use: <float> sphereSDF( in <float3> pos[], in <float> size] ) 
*/

#ifndef FNC_SPHERESDF
#define FNC_SPHERESDF
float sphereSDF(float3 p) { return length(p); }
float sphereSDF(float3 p, float s) { return sphereSDF(p) - s; }
#endif