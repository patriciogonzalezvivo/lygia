/*
contributors:  Inigo Quiles
description: elongate operation of two SDFs 
use: <float4> opElongate( in <float3> p, in <float3> h )
*/

#ifndef FNC_OPELONGATE
#define FNC_OPELONGATE

float2 opElongate( in float2 p, in float2 h ) {
    return p-clamp(p,-h,h); 
}

float3 opElongate( in float3 p, in float3 h ) {
    return p-clamp(p,-h,h); 
}

float4 opElongate( in float4 p, in float4 h ) {
    float3 q = abs(p)-h;
    return float4( max(q,0.0), min(max(q.x,max(q.y,q.z)), 0.0) );
}

#endif

