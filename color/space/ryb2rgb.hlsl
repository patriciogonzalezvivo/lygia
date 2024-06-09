#include "../../math/mmin.hlsl"
#include "../../math/mmax.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert from RYB to RGB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf
use: <float3|float4> ryb2rgb(<float3|float4> ryb)
examples:
  - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
  - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
  - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RYB2RGB
#define FNC_RYB2RGB

float3 ryb2rgb(float3 ryb) {
    float3 rgb = float3(0.0, 0.0, 0.0);
    
    // Remove white component
    float3 v = ryb - mmin(ryb);
    
    // Derive rgb
    float yb = min(v.y, v.b);
    rgb.r = v.r + v.y - yb;
    rgb.g = v.y + (2.0 * yb);
    rgb.b = 2.0 * (v.b - yb);
    
    // Normalize
    float n = mmax(rgb) / mmax(v);
    if (n > 0.0)
        rgb /= n;
    
    // Add black
    return rgb + mmin(1.0 - ryb);
}

float4 ryb2rgb(float4 ryb) { return float4(ryb2rgb(ryb.rgb), ryb.a); }

#endif