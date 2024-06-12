#include "../math/saturate.glsl"
#include "../sampler.glsl"

/*
contributors:
    - Matt DesLauriers
    - Johan Ismael
    - Patricio Gonzalez Vivo
description: Use LUT textures to modify colors (vec4 and vec3) or a position in a gradient (vec2 and floats)
use: lut(<SAMPLER_TYPE> texture, <vec4|vec3|vec2|float> value [, int row])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - LUT_N_ROWS: only useful on row LUTs to stack several of those one on top of each other
    - LUT_CELL_SIZE: cell side. DEfault. 32
    - LUT_SQUARE: the LUT have a SQQUARE shape and not just a long row
    - LUT_FLIP_Y: hen defined it expects a vertically flipled texture
examples:
    - /shaders/color_lut.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LUT
#define FNC_LUT

#ifdef LUT_SQUARE 

#ifdef LUT_FLIP_Y
#define SAMPLE2DCUBE_FLIP_Y
#endif

#ifndef SAMPLE_2DCUBE_CELLS_PER_SIDE
#ifdef LUT_CELLS_PER_SIDE
#define SAMPLE2DCUBE_CELLS_PER_SIDE LUT_CELLS_PER_SIDE
#else
#define SAMPLE2DCUBE_CELLS_PER_SIDE 8.0
#endif
#endif

#include "../sample/2DCube.glsl"
vec4 lut(in SAMPLER_TYPE tex_lut, in vec4 color, in int offset) { 
    return sample2DCube(tex_lut, color.rgb); 
}

#else

#ifndef LUT_N_ROWS
#define LUT_N_ROWS 1
#endif

#ifndef LUT_CELL_SIZE
#define LUT_CELL_SIZE 32.0
#endif

#ifndef LUT_CELLS_PER_SIDE
#define LUT_CELLS_PER_SIDE 8.0
#endif


// Data about how the LUTs rows are encoded
const float LUT_WIDTH = LUT_CELL_SIZE*LUT_CELL_SIZE;
const float LUT_OFFSET = 1./ float( LUT_N_ROWS );
const vec4 LUT_SIZE = vec4(LUT_WIDTH, LUT_CELL_SIZE, 1./LUT_WIDTH, 1./LUT_CELL_SIZE);

// Apply LUT to a COLOR
// ------------------------------------------------------------
vec4 lut(in SAMPLER_TYPE tex_lut, in vec4 color, in int offset) {
    vec3 scaledColor = clamp(color.rgb, vec3(0.), vec3(1.)) * (LUT_SIZE.y - 1.);
    float bFrac = fract(scaledColor.z);

    // offset by 0.5 pixel and fit within range [0.5, width-0.5]
    // to prevent bilinear filtering with adjacent colors
    vec2 texc = (.5 + scaledColor.xy) * LUT_SIZE.zw;

    // offset by the blue slice
    texc.x += (scaledColor.z - bFrac) * LUT_SIZE.w;
    texc.y *= LUT_OFFSET;
    texc.y += float(offset) * LUT_OFFSET;
    #ifndef LUT_FLIP_Y
    texc.y = 1. - texc.y; 
    #endif

    // sample the 2 adjacent blue slices
    vec4 b0 = SAMPLER_FNC(tex_lut, texc);
    vec4 b1 = SAMPLER_FNC(tex_lut, vec2(texc.x + LUT_SIZE.w, texc.y));

    // blend between the 2 adjacent blue slices
    color = mix(b0, b1, bFrac);

    return color;
}
#endif

vec4 lut(in SAMPLER_TYPE tex_lut, in vec4 color) { return lut(tex_lut, color, 0); }
vec3 lut(in SAMPLER_TYPE tex_lut, in vec3 color, in int offset) { return lut(tex_lut, vec4(color, 1.), offset).rgb; }
vec3 lut(in SAMPLER_TYPE tex_lut, in vec3 color) { return lut(tex_lut, color, 0).rgb; }

#endif