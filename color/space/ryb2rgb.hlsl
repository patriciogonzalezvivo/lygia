#include "../../math/mmin.hlsl"
#include "../../math/mmax.hlsl"
#include "../../math/cubicMix.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert from RYB to RGB color space. Based on http://nishitalab.org/user/UEI/publication/Sugita_IWAIT2015.pdf http://vis.computer.org/vis2004/DVD/infovis/papers/gossett.pdf
use: <float3|float4> ryb2rgb(<float3|float4> ryb)
options:
    - RYB_FAST: Use a non-homogeneous version of the conversion. Default is the homogeneous version.
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RYB_LERP 
#define RYB_LERP(A, B, t) cubicMix(A, B, t)
#endif

#ifndef FNC_RYB2RGB
#define FNC_RYB2RGB

#ifdef RYB_FAST

float3 ryb2rgb(float3 ryb) {
    // Remove the white from the color
    float w = mmin(ryb);
    ryb -= w;

    float max_y = mmax(ryb);
        
    // Get the green out of the yellow & blue
    float g = mmin(ryb.gb);
    float3 rgb = ryb - float3(0., g, g);
        
    if (rgb.b > 0. && g > 0.) {
        rgb.b *= 2.;
        g *= 2.;
    }

    // Redistribute the remaining yellow.
    rgb.r += rgb.g;
    rgb.g += g;

    // Normalize to values.
    float max_g = mmax(rgb);
    rgb *= (max_g > 0.) ? max_y / max_g : 1.;

    // Add the white back in.        
    return rgb + w;
}

#else

float3 ryb2rgb(float3 ryb) {
    const float3 ryb000 = float3(1., 1., 1.);       // white
    const float3 ryb001 = float3(.163, .373, .6);   // blue
    const float3 ryb010 = float3(1., 1., 0.);       // Yellow
    const float3 ryb100 = float3(1., 0., 0.);       // Red          
    const float3 ryb011 = float3(0., .66, .2);      // Green
    const float3 ryb101 = float3(.5, 0., .5);       // Violet
    const float3 ryb110 = float3(1., .5, 0.);       // Orange
    const float3 ryb111 = float3(0., 0., 0.);       // Black
    return RYB_LERP(RYB_LERP(
        RYB_LERP(ryb000, ryb001, ryb.z),
        RYB_LERP(ryb010, ryb011, ryb.z),
        ryb.y), RYB_LERP(
        RYB_LERP(ryb100, ryb101, ryb.z),
        RYB_LERP(ryb110, ryb111, ryb.z),
        ryb.y), ryb.x);
}

#endif

float4 ryb2rgb(float4 ryb) { return float4(ryb2rgb(ryb.rgb), ryb.a); }

#endif