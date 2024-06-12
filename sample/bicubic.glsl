#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: bicubic filter sampling
use: <vec4> sampleBicubic(<SAMPLER_TYPE> tex, <vec2> st, <vec2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
examples:
    - /shaders/sample_filter_bicubic.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLEBICUBIC
#define FNC_SAMPLEBICUBIC

vec4 sampleBicubic(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    vec4 o;
    o.x = s.x;
    o.y = s.y - 4.0 * s.x;
    o.z = s.z - 4.0 * s.y + 6.0 * s.x;
    o.w = 6.0 - o.x - o.y - o.z;
    return o;
}

vec4 sampleBicubic(SAMPLER_TYPE tex, vec2 st, vec2 texResolution) {
    vec2 pixel = 1.0 / texResolution;
    st = st * texResolution - 0.5;

    vec2 fxy = fract(st);
    st -= fxy;

    vec4 xcubic = sampleBicubic(fxy.x);
    vec4 ycubic = sampleBicubic(fxy.y);

    vec4 c = st.xxyy + vec2 (-0.5, 1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= pixel.xxyy;

    vec4 sample0 = SAMPLER_FNC(tex, offset.xz);
    vec4 sample1 = SAMPLER_FNC(tex, offset.yz);
    vec4 sample2 = SAMPLER_FNC(tex, offset.xw);
    vec4 sample3 = SAMPLER_FNC(tex, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix( mix(sample3, sample2, sx), 
                mix(sample1, sample0, sx), 
                sy);
}

#endif