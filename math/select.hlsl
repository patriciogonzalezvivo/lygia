/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns A when cond is true, and B otherwise. This is in part to bring a compatibility layer with WGSL 
use: <float|float2|float3|float4> select(<float|float2|float3|float4> A, <float|float2|float3|float4> B, <bool> cond)
*/

#ifndef FNC_SELECT
#define FNC_SELECT
int select(int A, int B, bool cond) { return cond ? A : B; }
float select(float A, float B, bool cond) { return cond ? A : B; }
float2 select(float2 A, float2 B, bool cond) { return cond ? A : B; }
float3 select(float3 A, float3 B, bool cond) { return cond ? A : B; }
float4 select(float4 A, float4 B, bool cond) { return cond ? A : B; }
#endif