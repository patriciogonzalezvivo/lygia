/*
Author: [Matt DesLauriers, Johan Ismael, Patricio Gonzalez Vivo]
description: Use LUT textures to modify colors (float4 and float3) or a position in a gradient (float2 and floats)
use: lut(<sampler2D> texture, <float4|float3|float2|float> value [, int row])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - LUT_N_ROWS: only useful on row LUTs to stack several of those one on top of each other 
    - LUT_CELL_SIZE: cell side. DEfault. 32
    - LUT_SQUARE: the LUT have a SQQUARE shape and not just a long row
    - LUT_FLIP_Y: hen defined it expects a vertically flipled texture 
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) tex2D(TEX, UV)
#endif

#ifndef LUT_N_ROWS
#define LUT_N_ROWS 1
#endif

#ifndef LUT_CELL_SIZE
#define LUT_CELL_SIZE 32.0
#endif

#ifndef FNC_LUT
#define FNC_LUT

#ifdef LUT_SQUARE 
float4 lut(in sampler2D tex_lut, in float4 color, in int offset) {
    float blueColor = color.b * 63.0;

    const float pixel = 1.0/512.0;
    const float halt_pixel = pixel * 0.5;

    float2 quad1 = float2(0.0, 0.0);
    quad1.y = floor(floor(blueColor) / 8.0);
    quad1.x = floor(blueColor) - (quad1.y * 8.0);
    
    float2 quad2 = float2(0.0, 0.0);
    quad2.y = floor(ceil(blueColor) / 8.0);
    quad2.x = ceil(blueColor) - (quad2.y * 8.0);
    
    float2 texPos1 = (quad1 * 0.125) + halt_pixel + ((0.125 - pixel) * color.rg);
    texPos1 = saturate(texPos1);

    #ifdef LUT_FLIP_Y
    texPos1.y = 1.0-texPos1.y;
    #endif
    
    float2 texPos2 = (quad2 * 0.125) + halt_pixel + ((0.125 - pixel) * color.rg);
    texPos2 = saturate(texPos2);

    #ifdef LUT_FLIP_Y
    texPos2.y = 1.0-texPos2.y;
    #endif
    
    float4 b0 = SAMPLER_FNC(tex_lut, texPos1);
    float4 b1 = SAMPLER_FNC(tex_lut, texPos2);

    return lerp(b0, b1, saturate(blueColor));
}

#else
// Data about how the LUTs rows are encoded
const float LUT_WIDTH = LUT_CELL_SIZE*LUT_CELL_SIZE;
const float LUT_OFFSET = 1./ float( LUT_N_ROWS);
const float4 LUT_SIZE = float4(LUT_WIDTH, LUT_CELL_SIZE, 1./LUT_WIDTH, 1./LUT_CELL_SIZE);

// Apply LUT to a COLOR
// ------------------------------------------------------------
float4 lut(in sampler2D tex_lut, in float4 color, in int offset) {
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
    float4 b0 = SAMPLER_FNC(tex_lut, texc);
    float4 b1 = SAMPLER_FNC(tex_lut, float2(texc.x + LUT_SIZE.w, texc.y));

    // blend between the 2 adjacent blue slices
    color = lerp(b0, b1, bFrac);

    return color;
}
#endif

float4 lut(in sampler2D tex_lut, in float4 color) { return lut(tex_lut, color, 0); }

float3 lut(in sampler2D tex_lut, in float3 color, in int offset) { return lut(tex_lut, float4(color, 1.), offset).rgb; }
float3 lut(in sampler2D tex_lut, in float3 color) { return lut(tex_lut, color, 0).rgb; }

#endif