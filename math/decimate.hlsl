/*
contributors: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion
use: <float|float2|float3|float4> decimate(<float|float2|float3|float4> value, <float|float2|float3|float4> presicion)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
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