/*
original_author:  Inigo Quiles
description: onion operation of one SDFs 
use: <float4> opElongate( in <float3> p, in <float3> h )
*/

#ifndef FNC_OPONION
#define FNC_OPONION

float opOnion( in float d, in float h ) {
    return abs(d)-h;
}

#endif

