/*
original_author: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion 
use: decimation(<float|float2|float3|float4> value, <float|float2|float3|float4> presicion)
*/

#ifndef FNC_DECIMATION
#define FNC_DECIMATION
#define decimation(value, presicion) (floor(value * presicion)/presicion)
#endif