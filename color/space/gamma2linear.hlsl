/*
contributors: Patricio Gonzalez Vivo
description: Convert from gamma to linear color space.
use: gamma2linear(<float|float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#if !defined(GAMMA) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
#define GAMMA 2.2
#endif

#ifndef FNC_GAMMA2LINEAR
#define FNC_GAMMA2LINEAR
float gamma2linear(in float v) {
#ifdef GAMMA
    return pow(v, GAMMA);
#else
    // assume gamma 2.0
    return v * v;
#endif
}

float3 gamma2linear(in float3 v) {
#ifdef GAMMA
    return pow(v, float3(GAMMA, GAMMA, GAMMA));
#else
    // assume gamma 2.0
    return v * v;
#endif
}

float4 gamma2linear(in float4 v) {
    return float4(gamma2linear(v.rgb), v.a);
}
#endif
