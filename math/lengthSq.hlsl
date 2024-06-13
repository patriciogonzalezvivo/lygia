/*
contributors: Patricio Gonzalez Vivo
description: Squared length
use: <float2|float3|float4> lengthSq(<float2|float3|float4> v)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LENGTHSQ
#define FNC_LENGTHSQ

float lengthSq(in float2 v) { return dot(v, v); }
float lengthSq(in float3 v) { return dot(v, v); }
float lengthSq(in float4 v) { return dot(v, v); }

#endif
