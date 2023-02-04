/*
original_author: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion 
use: decimate(<float|float2|float3|float4> value, <float|float2|float3|float4> presicion)
*/

#ifndef FNC_DECIMATE
#define FNC_DECIMATE

#define decimate(value, presicion) (floor(value * presicion)/presicion)

#endif