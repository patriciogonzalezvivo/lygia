/*
contributors:  Inigo Quiles
description: generate the SDF of a link
use: <float> linkSDF( <float3> p, <float> le, <float> r1, <float> r2 ) 
*/

#ifndef FNC_LINKSDF
#define FNC_LINKSDF
float linkSDF( float3 p, float le, float r1, float r2 ) {
    float3 q = float3( p.x, max(abs(p.y)-le,0.0), p.z );
    return length(float2(length(q.xy)-r1,q.z)) - r2;
}
#endif