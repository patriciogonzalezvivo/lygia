/*
contributors:  Inigo Quiles
description: generate the SDF of a plane
use: <float> planeSDF( in <float3> pos, in <float2> h ) 
*/

#ifndef FNC_PLANESDF
#define FNC_PLANESDF
float planeSDF( float3 p ) { 
   return p.y; 
}

float planeSDF(float3 p, float3 planePoint, float3 planeNormal) {
    return (dot(planeNormal, p) + dot(planeNormal, planePoint)) / length(planeNormal);
}
#endif