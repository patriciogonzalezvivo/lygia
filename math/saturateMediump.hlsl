/*
contributors: Patricio Gonzalez Vivo
description: clamp a value between 0 and the medium precision max (65504.0) for floating points
use: <float|float2|float3|float4> saturateMediump(<float|float2|float3|float4> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SATURATEMEDIUMP
#define FNC_SATURATEMEDIUMP

#ifndef MEDIUMP_FLT_MAX
#define MEDIUMP_FLT_MAX    65504.0
#endif

#if defined(TARGET_MOBILE) || defined(PLATFORM_WEBGL) || defined(PLATFORM_RPI)
#define saturateMediump(x) min(x, MEDIUMP_FLT_MAX)
#else
#define saturateMediump(x) x
#endif

#endif