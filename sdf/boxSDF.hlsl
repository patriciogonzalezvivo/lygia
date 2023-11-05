/*
contributors:  Inigo Quiles
description: generate the SDF of a box
use: <float> boxSDF( in <float3> pos [, in <float3> borders ] ) 
*/

#ifndef FNC_BOXSDF
#define FNC_BOXSDF

float boxSDF( float3 p ) {
    float3 d = abs(p);
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float boxSDF( float3 p, float3 b ) {
    float3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

#endif