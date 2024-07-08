#include "../sampler.hlsl"

/*
contributors:
    - Matt DesLauriers
    - Johan Ismael
    - Patricio Gonzalez Vivo
description: Use LUT textures to modify colors (float4 and float3) or a position in a gradient (float2 and floats)
use: lut(<SAMPLER_TYPE> texture, <float4|float3|float2|float> value [, int row])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - LUT_N_ROWS: only useful on row LUTs to stack several of those one on top of each other
    - LUT_CELL_SIZE: cell side. DEfault. 32
    - LUT_SQUARE: the LUT have a SQQUARE shape and not just a long row
    - LUT_FLIP_Y: hen defined it expects a vertically flipled texture
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LUT
#define FNC_LUT

#ifdef LUT_SQUARE 

#ifdef LUT_FLIP_Y
#define SAMPLE_2DCUBE_FLIP_Y
#endif

#ifndef SAMPLE_2DCUBE_CELLS_PER_SIDE
#ifdef LUT_CELLS_PER_SIDE
#define SAMPLE_2DCUBE_CELLS_PER_SIDE LUT_CELLS_PER_SIDE
#else
#define SAMPLE_2DCUBE_CELLS_PER_SIDE 8.0
#endif
#endif

#include "../sample/2DCube.hlsl"
float4 lut(in SAMPLER_TYPE tex, in float4 color, in int offset) { 
    return sample2DCube(tex, color.rgb); 
}

#else
// Data about how the LUTs rows are encoded
static const float LUT_WIDTH = LUT_CELL_SIZE*LUT_CELL_SIZE;
static const float LUT_OFFSET = 1./ float( LUT_N_ROWS);
static const float4 LUT_SIZE = float4(LUT_WIDTH, LUT_CELL_SIZE, 1./LUT_WIDTH, 1./LUT_CELL_SIZE);

// Apply LUT to a COLOR
// ------------------------------------------------------------
float4 lut(in SAMPLER_TYPE tex, in float4 color, in int offset) {
    float3 scaledColor = clamp(color.rgb, float3(0., 0., 0.), float3(1., 1., 1.)) * (LUT_SIZE.y - 1.);
    float bFrac = frac(scaledColor.z);

    // offset by 0.5 pixel and fit within range [0.5, width-0.5]
    // to prevent bilinear filtering with adjacent colors
    float2 texc = (.5 + scaledColor.xy) * LUT_SIZE.zw;

    // offset by the blue slice
    texc.x += (scaledColor.z - bFrac) * LUT_SIZE.w;
    texc.y *= LUT_OFFSET;
    texc.y += float(offset) * LUT_OFFSET;
    #ifndef LUT_FLIP_Y
    texc.y = 1. - texc.y; 
    #endif

    // sample the 2 adjacent blue slices
    float4 b0 = SAMPLER_FNC(tex, texc);
    float4 b1 = SAMPLER_FNC(tex, float2(texc.x + LUT_SIZE.w, texc.y));

    // blend between the 2 adjacent blue slices
    color = lerp(b0, b1, bFrac);

    return color;
}
#endif

float4 lut(in SAMPLER_TYPE tex, in float4 color) { return lut(tex, color, 0); }
float3 lut(in SAMPLER_TYPE tex, in float3 color, in int offset) { return lut(tex, float4(color, 1.), offset).rgb; }
float3 lut(in SAMPLER_TYPE tex, in float3 color) { return lut(tex, color, 0).rgb; }

#endif