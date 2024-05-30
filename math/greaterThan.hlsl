/*
contributors: [Stefan Gustavson, Ian McEwan]
description: greaterThan, returns 1 if x > y, 0 otherwise
use: greaterThan(<float|float2|float3|float4> x, y)
*/

#ifndef FNC_GREATERTHAN
#define FNC_GREATERTHAN
float greaterThan(float x, float y) { return step(y, x); }
float2 greaterThan(float2 x, float2 y) { return step(y, x); }
float3 greaterThan(float3 x, float3 y) { return step(y, x); }
float4 greaterThan(float4 x, float4 y) { return step(y, x); }
#endif
