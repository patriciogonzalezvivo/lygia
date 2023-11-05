/*
contributors:  Inigo Quiles
description: generate a SDF of a capsule
use: <float> capusleSDF( in <float3> pos, in <float3> a, <float3> b, <float> r ) 
*/

#ifndef FNC_CAPSULESDF
#define FNC_CAPSULESDF

float capsuleSDF( float3 p, float3 a, float3 b, float r ) {
    float3 pa = p-a, ba = b-a;
    float h = saturate( dot(pa,ba)/dot(ba,ba) );
    return length( pa - ba*h ) - r;
}

#endif