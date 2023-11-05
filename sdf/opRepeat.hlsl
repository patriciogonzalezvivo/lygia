#include "../math/mod.hlsl"

/*
contributors:  Inigo Quiles
description: repite operation of one 2D SDFs 
use: <float4> opElongate( in <float3> p, in <float3> h )
*/

#ifndef FNC_OPREPEAT
#define FNC_OPREPEAT

float2 opRepeat( in float2 p, in float s ) {
    return mod(p+s*0.5,s)-s*0.5;
}

float3 opRepeat( in float3 p, in float3 c ) {
    return mod(p+0.5*c,c)-0.5*c;
}

float2 opRepeat( in float2 p, in float2 lima, in float2 limb, in float s ) {
    return p-s*clamp(floor(p/s), lima, limb);
}

float3 opRepeat( in float3 p, in float3 lima, in float3 limb, in float s ) {
    return p-s*clamp(floor(p/s), lima, limb);
}

#endif

