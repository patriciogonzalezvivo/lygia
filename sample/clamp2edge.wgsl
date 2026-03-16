#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a clamp to edge texture
use: <vec4> sampleClamp2edge(<SAMPLER_TYPE> tex, <vec2> st [, <vec2> texResolution]);
options:
    - SAMPLER_FNC(TEX, UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleClamp2edge(tex: SAMPLER_TYPE, st: vec2f, texResolution: vec2f) -> vec4f {
    let pixel = 1.0/texResolution;
    return SAMPLER_FNC( tex, clamp(st, pixel, 1.0-pixel) );
}

fn sampleClamp2edgea(tex: SAMPLER_TYPE, st: vec2f) -> vec4f {
    return SAMPLER_FNC( tex, clamp(st, vec2f(0.01), vec2f(0.99) ) ); 
}

fn sampleClamp2edgeb(tex: SAMPLER_TYPE, st: vec2f, edge: f32) -> vec4f {
    return SAMPLER_FNC( tex, clamp(st, vec2f(edge), vec2f(1.0 - edge) ) ); 
}
