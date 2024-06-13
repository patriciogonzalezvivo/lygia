#include "../sampler.hlsl"
#include "../math/const.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: fill alpha with edge colors
use: <float4> fillAlpha(<SAMPLER_TYPE> texture, <float2> st, <float2> pixel, <int> passes)
options:
    - SAMPLER_FNC(TEX, UV)
    - ALPHAFILL_RADIUS
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef ALPHAFILL_RADIUS
#define ALPHAFILL_RADIUS 2.0
#endif

#ifndef FNC_ALPHAFILL
#define FNC_ALPHAFILL

float4 alphaFill(SAMPLER_TYPE tex, float2 st, float2 pixel, int passes) {
    float4 accum = float4(0.0, 0.0, 0.0, 0.0);
    float max_dist = sqrt(ALPHAFILL_RADIUS * ALPHAFILL_RADIUS);
    for (int s = 0; s < passes; s++) {    
        float2 spiral = float2(sin(float(s)*GOLDEN_ANGLE), cos(float(s)*GOLDEN_ANGLE));
        float dist = sqrt(ALPHAFILL_RADIUS * float(s));
        spiral *= dist;
        float4 sampled_pixel = SAMPLER_FNC(tex, st + spiral * pixel);
        sampled_pixel.rgb *= sampled_pixel.a;
        accum += sampled_pixel * (1.0 / (1.0 + dist));
        if (accum.a >= 1.0) 
            break;
    }

    return accum.rgba / max(0.0001, accum.a);
}

#endif