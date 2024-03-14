/*
contributors: Patricio Gonzalez Vivo
description: extend HLSL Max function to add more arguments
use: 
  - <float> mmax(<float> A, <float> B, <float> C[, <float> D])
  - <float2|float3|float4> mmax(<float2|float3|float4> A)
*/

#ifndef FNC_MMAX
#define FNC_MMAX

float mmax(in float a, in float b) { return max(a, b); }
float mmax(in float a, in float b, in float c) { return max(a, max(b, c)); }
float mmax(in float a, in float b, in float c, in float d) { return max(max(a, b), max(c, d)); }

float mmax(const float2 v) { return max(v.x, v.y); }
float mmax(const float3 v) { return mmax(v.x, v.y, v.z); }
float mmax(const float4 v) { return mmax(v.x, v.y, v.z, v.w); }

#endif
