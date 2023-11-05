/*
contributors:  Inigo Quiles
description: generate the SDF of a bounding box
use: <float> boxFrameSDF( <float3> p, <float3> b, <float> e )
*/

#ifndef FNC_BOXFRAMESDF
#define FNC_BOXFRAMESDF

float boxFrameSDF( float3 p, float3 b, float e ) {
    p = abs(p) - b;
    float3 q = abs(p + e) - e;

    return min(min(
        length(max(float3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
        length(max(float3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
        length(max(float3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

#endif