/*
contributors: Patricio Gonzalez Vivo
description: Index of refraction to ratio of index of refraction
use: <float|float3|float4> ior2eta(<float|float3|float4> ior)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_IOR2ETA
#define FNC_IOR2ETA
float ior2eta( const float ior ) { return 1.0/ior; }
float3 ior2eta( const float3 ior ) { return 1.0/ior; }
float4 ior2eta( const float4 ior ) { return float4(1.0/ior.rgb, ior.a); }
#endif