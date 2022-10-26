#include "boxSDF.glsl"

/*
description: generate the SDF of a cube
use: <float> cubeSDF( in <vec3> pos, in <float> size ) 
*/

#ifndef FNC_CUBESDF
#define FNC_CUBESDF
float cubeSDF(vec3 p, float s)  { return boxSDF(p, vec3(s)); }
#endif