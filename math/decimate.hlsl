/*
contributors: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion 
use: <float|float2|float3|float4> decimate(<float|float2|float3|float4> value, <float|float2|float3|float4> presicion)
*/

#ifndef FNC_DECIMATE
#define FNC_DECIMATE
float decimate(float v, float p){ return floor(v*p)/p; }
float2 decimate(float2 v, float p){ return floor(v*p)/p; }
float2 decimate(float2 v, float2 p){ return floor(v*p)/p; }
float3 decimate(float3 v, float p){ return floor(v*p)/p; }
float3 decimate(float3 v, float3 p){ return floor(v*p)/p; }
float4 decimate(float4 v, float p){ return floor(v*p)/p; }
float4 decimate(float4 v, float4 p){ return floor(v*p)/p; }
#endif