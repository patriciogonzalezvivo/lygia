#include "digits.hlsl"
#include "../sample/nearest.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Given a texture and a coordinate draw the nearest pixel 
use: <float4> colorPicker(SAMPLER_TYPE tex, <float2> pos, <float2> texResolution, <float2> st) 
options:
    DIGITS_DECIMALS: number of decimals after the point, defaults to 2
    DIGITS_SIZE: size of the font, defaults to float2(.025)
    PIXEL_SIZE: size of the pixel to sample
    PIXEL_KERNEL_SIZE: kernels size of pixels to sample
    PIXEL_SAMPLER_FNC: sampler function to use. Default is nearest
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


#ifndef PIXEL_SAMPLER_FNC
#define PIXEL_SAMPLER_FNC(TEX, UV) sampleNearest(TEX, UV, texResolution)
#endif

#ifndef PIXEL_KERNEL_SIZE 
#define PIXEL_KERNEL_SIZE 1
#endif

#ifndef PIXEL_SIZE 
#define PIXEL_SIZE float2(0.025, 0.025)
#endif

#ifndef FNC_PIXEL
#define FNC_PIXEL
float4 colorPicker(SAMPLER_TYPE tex, in float2 pos, float2 texResolution, in float2 st) {
    float4 rta = float4(0.0, 0.0, 0.0, 0.0);

    float2 t_size = float(PIXEL_KERNEL_SIZE * 2 + 1) * PIXEL_SIZE;
    float2 v_size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET);
    t_size.x = max(t_size.x, v_size.x * 2.0);
    t_size.y += DIGITS_SIZE.y * 2.0;
    float2 pixel = 1.0/texResolution;
    float4 val = PIXEL_SAMPLER_FNC(tex, pos);

    // draw kernel
    for (int x = -PIXEL_KERNEL_SIZE; x <= PIXEL_KERNEL_SIZE; x++) {
        for (int y = -PIXEL_KERNEL_SIZE; y <= PIXEL_KERNEL_SIZE; y++) {
            float2 o = float2(float(x), float(y));
            float2 st_o = st - o * PIXEL_SIZE * 2.0;
            rta += PIXEL_SAMPLER_FNC(tex, pos + o * pixel) * step(-PIXEL_SIZE.x, st_o.x) * step(st_o.x, PIXEL_SIZE.x) * step(-PIXEL_SIZE.y, st_o.y) * step(st_o.y, PIXEL_SIZE.y);
        }
    }

    rta.a = max(rta.a, 0.5 * step(-t_size.x, st.x) * step(st.x, t_size.x) * step(-t_size.y, st.y) * step(st.y, t_size.y));
    
    float2 st_v = st + t_size * float2(0.0, 1.0) + v_size * float2(2.0, 0.0) - DIGITS_SIZE * float2(0.0, 0.5);
    rta += digits(st_v, val);

    float2 st_c = st - t_size * float2(0.5, 1.0) + v_size * float2(2.0, 0.5);
    rta += digits(st_c, pos);

    return rta;
}

#endif