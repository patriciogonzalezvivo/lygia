#include "../../math/mmin.hlsl"
#include "../../math/mmax.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a color from RGB to RYB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <float3> ryb2rgb(<float3> ryb)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RGB2RYB
#define FNC_RGB2RYB

float3 rgb2ryb(float3 rgb) {
    // Remove white component
    float3 v = rgb - mmin(rgb);

    // Derive ryb
    float3 ryb = float3(0.0, 0.0, 0.0);
    float rg = min(v.r, v.g);
    ryb.r = v.r - rg;
    ryb.y = 0.5 * (v.g + rg);
    ryb.b = 0.5 * (v.b + v.g - rg);
    
    // Normalize
    float n = mmax(ryb) / mmax(v);
    if (n > 0.0)
    	ryb /= n;
    
    // Add black 
    return ryb + mmin(1.0 - rgb);
}

float4 rgb2ryb(float4 rgb) { return float4(rgb2ryb(rgb.rgb), rgb.a); }

#endif
