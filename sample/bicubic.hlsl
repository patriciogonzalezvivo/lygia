#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: bicubic filter sampling
use: <float4> sampleBicubic(<SAMPLER_TYPE> tex, <float2> st, <float2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLEBICUBIC
#define FNC_SAMPLEBICUBIC

float4 sampleBicubic(float v) {
    float4 n = float4(1.0, 2.0, 3.0, 4.0) - v;
    float4 s = n * n * n;
    float4 o;
    o.x = s.x;
    o.y = s.y - 4.0 * s.x;
    o.z = s.z - 4.0 * s.y + 6.0 * s.x;
    o.w = 6.0 - o.x - o.y - o.z;
    return o;
}

float4 sampleBicubic(SAMPLER_TYPE tex, float2 st, float2 texResolution) {
    float2 pixel = 1.0 / texResolution;
    st = st * texResolution - 0.5;

    float2 fxy = frac(st);
    st -= fxy;

    float4 xcubic = sampleBicubic(fxy.x);
    float4 ycubic = sampleBicubic(fxy.y);

    float4 c = st.xxyy + float2 (-0.5, 1.5).xyxy;

    float4 s = float4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    float4 offset = c + float4(xcubic.yw, ycubic.yw) / s;

    offset *= pixel.xxyy;

    float4 sample0 = SAMPLER_FNC(tex, offset.xz);
    float4 sample1 = SAMPLER_FNC(tex, offset.yz);
    float4 sample2 = SAMPLER_FNC(tex, offset.xw);
    float4 sample3 = SAMPLER_FNC(tex, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return lerp(    lerp(sample3, sample2, sx), 
                    lerp(sample1, sample0, sx), 
                    sy);
}

#endif