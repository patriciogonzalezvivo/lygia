#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Make a radial blur, with dir as the direction to the center and strength as the amount
use: radialBlur(<SAMPLER_TYPE> texture, <float2> st, <float2> dir [, <float> strength])
options:
    - RADIALBLUR_KERNELSIZE: Default 64
    - RADIALBLUR_STRENGTH: Default 0.125
    - RADIALBLUR_TYPE: Default `float4`
    - RADIALBLUR_SAMPLER_FNC(TEX, UV): Default `tex2D(tex, TEX, UV)`
    - SAMPLER_FNC(TEX, UV): optional depending the target version of HLSL
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RADIALBLUR_KERNELSIZE
#define RADIALBLUR_KERNELSIZE 64
#endif

#ifndef RADIALBLUR_STRENGTH
#define RADIALBLUR_STRENGTH .125
#endif

#ifndef RADIALBLUR_TYPE
#define RADIALBLUR_TYPE float4
#endif

#ifndef RADIALBLUR_SAMPLER_FNC
#define RADIALBLUR_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_RADIALBLUR
#define FNC_RADIALBLUR
RADIALBLUR_TYPE radialBlur(in SAMPLER_TYPE tex, in float2 st, in float2 dir, in float strength) {
    RADIALBLUR_TYPE color = float4(0.0, 0.0, 0.0, 0.0);
    float f_samples = float(RADIALBLUR_KERNELSIZE);
    float f_factor = 1./f_samples;
    for (int i = 0; i < RADIALBLUR_KERNELSIZE; i += 2) {
        color += RADIALBLUR_SAMPLER_FNC(tex, st + float(i) * f_factor * dir * strength);
        color += RADIALBLUR_SAMPLER_FNC(tex, st + float(i+1) * f_factor * dir * strength);
    }
    return color * f_factor;
}

RADIALBLUR_TYPE radialBlur(in SAMPLER_TYPE tex, in float2 st, in float2 dir) {
    return radialBlur(tex, st, dir, RADIALBLUR_STRENGTH);
}
#endif
