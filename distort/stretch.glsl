/*
author: Patricio Gonzalez Vivo, Johan Ismael
description: Samples multiple times a texture in the specified direction
use: stretch(<sampler2D> tex, <vec2> st, <vec2> direction [, int samples])
options:
    STRETCH_SAMPLES: number of samples taken, defaults to 20
    STRETCH_TYPE: return type, defauls to vec4
    STRETCH_SAMPLER_FNC(POS_UV): function used to sample the input texture, defaults to texture2D(tex, POS_UV)
    STRETCH_WEIGHT: shaping equation to multiply the sample weight.
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo and Johan Ismael
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOF
*/

#ifndef STRETCH_SAMPLES
#define STRETCH_SAMPLES 20
#endif

#ifndef STRETCH_TYPE
#define STRETCH_TYPE vec4
#endif

#ifndef STRETCH_SAMPLER_FNC
#define STRETCH_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
#endif

#ifndef FNC_STRETCH
#define FNC_STRETCH
STRETCH_TYPE stretch(in sampler2D tex, in vec2 st, in vec2 direction, const int i_samples) {
    float f_samples = float(i_samples);
    STRETCH_TYPE color = STRETCH_TYPE(0.);

    #ifdef PLATFORM_WEBGL
    for (int i = 0; i < 50; i++) {
        if (i == i_samples) break;
    #else
    for (int i = 0; i < i_samples; i++) {
    #endif

        float f_sample = float(i);
        STRETCH_TYPE tx = STRETCH_SAMPLER_FNC(st + direction * f_sample);
        #ifdef STRETCH_WEIGHT
        tx *= STRETCH_WEIGHT;
        #endif
        color += tx;
    }
    return color / f_samples;
}

STRETCH_TYPE stretch(in sampler2D tex, in vec2 st, in vec2 direction) {
    float f_samples = float(STRETCH_SAMPLES);  
    STRETCH_TYPE color = STRETCH_TYPE(0.);
    for (int i = 0; i < STRETCH_SAMPLES; i++) {
        float f_sample = float(i);    
        STRETCH_TYPE tx = STRETCH_SAMPLER_FNC(st + direction * f_sample);
        #ifdef STRETCH_WEIGHT
        tx *= STRETCH_WEIGHT;    
        #endif
        color += tx;
    }
    return color / f_samples;
}
#endif
