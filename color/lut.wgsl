#include "../math/saturate.wgsl"
#include "../sampler.wgsl"

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

// #define SAMPLE2DCUBE_FLIP_Y

// #define SAMPLE2DCUBE_CELLS_PER_SIDE LUT_CELLS_PER_SIDE
const SAMPLE2DCUBE_CELLS_PER_SIDE: f32 = 8.0;

#include "../sample/2DCube.wgsl"
fn lut(tex_lut: SAMPLER_TYPE, color: vec4f, offset: i32) -> vec4f {
    return sample2DCube(tex_lut, color.rgb); 
}

const LUT_N_ROWS: f32 = 1;

const LUT_CELL_SIZE: f32 = 32.0;

const LUT_CELLS_PER_SIDE: f32 = 8.0;

// Data about how the LUTs rows are encoded
let LUT_WIDTH = LUT_CELL_SIZE*LUT_CELL_SIZE;
let LUT_OFFSET = 1./ float( LUT_N_ROWS );
let LUT_SIZE = vec4f(LUT_WIDTH, LUT_CELL_SIZE, 1./LUT_WIDTH, 1./LUT_CELL_SIZE);

// Apply LUT to a COLOR
// ------------------------------------------------------------
fn luta(tex_lut: SAMPLER_TYPE, color: vec4f, offset: i32) -> vec4f {
    let scaledColor = clamp(color.rgb, vec3f(0.), vec3f(1.)) * (LUT_SIZE.y - 1.);
    let bFrac = fract(scaledColor.z);

    // offset by 0.5 pixel and fit within range [0.5, width-0.5]
    // to prevent bilinear filtering with adjacent colors
    let texc = (.5 + scaledColor.xy) * LUT_SIZE.zw;

    // offset by the blue slice
    texc.x += (scaledColor.z - bFrac) * LUT_SIZE.w;
    texc.y *= LUT_OFFSET;
    texc.y += float(offset) * LUT_OFFSET;
    texc.y = 1. - texc.y; 

    // sample the 2 adjacent blue slices
    let b0 = SAMPLER_FNC(tex_lut, texc);
    let b1 = SAMPLER_FNC(tex_lut, vec2f(texc.x + LUT_SIZE.w, texc.y));

    // blend between the 2 adjacent blue slices
    color = mix(b0, b1, bFrac);

    return color;
}

fn lutb(tex_lut: SAMPLER_TYPE, color: vec4f) -> vec4f { return lut(tex_lut, color, 0); }
fn lutc(tex_lut: SAMPLER_TYPE, color: vec3f, offset: i32) -> vec3f { return lut(tex_lut, vec4f(color, 1.), offset).rgb; }
fn lutd(tex_lut: SAMPLER_TYPE, color: vec3f) -> vec3f { return lut(tex_lut, color, 0).rgb; }
