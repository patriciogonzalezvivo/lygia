#include "shadow.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample shadow map using PCF
use: <float> sampleShadowLerp(<SAMPLER_TYPE> depths, <float2> size, <float2> uv, <float> compare)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLESHADOWLERP
#define FNC_SAMPLESHADOWLERP

float sampleShadowLerp(SAMPLER_TYPE depths, float2 size, float2 uv, float compare) {
    float2 texelSize = 1.0/size;
    float2 f = frac(uv*size+0.5);
    float2 centroidUV = floor(uv*size+0.5)/size;
    float lb = sampleShadow(depths, centroidUV+texelSize*float2(0.0, 0.0), compare);
    float lt = sampleShadow(depths, centroidUV+texelSize*float2(0.0, 1.0), compare);
    float rb = sampleShadow(depths, centroidUV+texelSize*float2(1.0, 0.0), compare);
    float rt = sampleShadow(depths, centroidUV+texelSize*float2(1.0, 1.0), compare);
    float a = lerp(lb, lt, f.y);
    float b = lerp(rb, rt, f.y);
    float c = lerp(a, b, f.x);
    return c;
}

#endif