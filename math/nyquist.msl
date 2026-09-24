/*
contributors: Shadi El Hajj
description: A simple low-pass filter which attenuates high-frequencies.
use: nyquist(<float> value, <float> fwidth)
use: nyquist(<float> value, <float> width, <float> strength)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_NYQUIST
#define FNC_NYQUIST

#ifndef NYQUIST_FILTER_CENTER
#define NYQUIST_FILTER_CENTER 0.5
#endif

#ifndef NYQUIST_FILTER_WIDTH
#define NYQUIST_FILTER_WIDTH 0.25
#endif

float nyquist(float x, float width){
    float cutoffStart = NYQUIST_FILTER_CENTER - NYQUIST_FILTER_WIDTH;
    float cutoffEnd = NYQUIST_FILTER_CENTER + NYQUIST_FILTER_WIDTH;
    float f = smoothstep(cutoffEnd, cutoffStart, width);
    return mix(0.5, x, f);
}

#endif
