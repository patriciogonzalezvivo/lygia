#include "boxSDF.hlsl"

/*
description: generate the SDF of a cube
use: <float> cubeSDF( in <float3> pos, in <float> size ) 
*/

#ifndef FNC_CUBESDF
#define FNC_CUBESDF
float cubeSDF(float3 p, float s)  { return boxSDF(p, float3(s, s, s)); }
#endif