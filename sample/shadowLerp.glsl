#include "shadow.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample shadow map using PCF
use: <float> sampleShadowLerp(<SAMPLER_TYPE> depths, <vec2> size, <vec2> uv, <float> compare)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLESHADOWLERP
#define FNC_SAMPLESHADOWLERP

float sampleShadowLerp(SAMPLER_TYPE depths, vec2 size, vec2 uv, float compare) {
    vec2 texelSize = vec2(1.0)/size;
    vec2 f = fract(uv*size+0.5);
    vec2 centroidUV = floor(uv*size+0.5)/size;
    float lb = sampleShadow(depths, centroidUV+texelSize*vec2(0.0, 0.0), compare);
    float lt = sampleShadow(depths, centroidUV+texelSize*vec2(0.0, 1.0), compare);
    float rb = sampleShadow(depths, centroidUV+texelSize*vec2(1.0, 0.0), compare);
    float rt = sampleShadow(depths, centroidUV+texelSize*vec2(1.0, 1.0), compare);
    float a = mix(lb, lt, f.y);
    float b = mix(rb, rt, f.y);
    float c = mix(a, b, f.x);
    return c;
}

#endif