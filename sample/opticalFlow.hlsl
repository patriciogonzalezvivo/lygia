#include "../sampler.hlsl"
#include "../math/const.hlsl"
#include "../color/space/rgb2hsv.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    sample an optical flow direction where the angle is encoded in the Hue and magnitude in the Saturation
use: sampleFlow(<SAMPLER_TYPE> tex, <float2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


#ifndef SAMPLEOPTICALFLOW_SAMPLE_FNC
#define SAMPLEOPTICALFLOW_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb
#endif

#ifndef FNC_SAMPLEOPTICALFLOW
#define FNC_SAMPLEOPTICALFLOW
float2 sampleOpticalFlow(SAMPLER_TYPE tex, float2 st) {
    float3 data = SAMPLEOPTICALFLOW_SAMPLE_FNC(tex, st);
    float3 hsv = rgb2hsv(data.rgb);
    float a = (hsv.x * 2.0 - 1.0) * PI;
    return float2(cos(a), -sin(a)) * hsv.y;
}

float2 sampleOpticalFlow(SAMPLER_TYPE tex, float2 st, float2 resolution, float max_dist) { return sampleOpticalFlow(tex, st) * (max_dist / resolution); }
#endif