#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Sharpening filter
use: sharpen(<SAMPLER_TYPE> texture, <float2> st, <float2> renderSize [, float streanght])
options:
    - SHARPEN_KERNELSIZE: Defaults 2
    - SHARPEN_TYPE: defaults to float3
    - SHARPEN_SAMPLER_FNC(TEX, UV): defaults to texture2D(TEX, UV).rgb
    - SHARPEN_FNC: defaults to sharpenFast
    - SAMPLER_FNC(TEX, UV): optional depending the target version of HLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RADIALBLUR_SAMPLER_FNC
#define RADIALBLUR_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef SHARPEN_TYPE
#define SHARPEN_TYPE float3
#endif

#ifndef SHARPEN_SAMPLER_FNC
#define SHARPEN_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb
#endif

#ifndef SHARPEN_FNC
#define SHARPEN_FNC sharpenFast
#endif

#include "sharpen/fast.hlsl"
#include "sharpen/adaptive.hlsl"
#include "sharpen/contrastAdaptive.hlsl"

#ifndef FNC_SHARPEN
#define FNC_SHARPEN

SHARPEN_TYPE sharpen(in SAMPLER_TYPE tex, in float2 st, in float2 pixel, float strenght) {
    return SHARPEN_FNC(tex, st, pixel, strenght);
}

SHARPEN_TYPE sharpen(in SAMPLER_TYPE tex, in float2 st, in float2 pixel) {
    return SHARPEN_FNC(tex, st, pixel);
}

#endif 