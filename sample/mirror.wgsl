#include "../sampler.wgsl"
#include "../math/mirror.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a mirror wrapping texture
use: <vec4> sampleMirror(<SAMPLER_TYPE> tex, <vec2> st);
options:
    - SAMPLER_FNC(TEX, UV)
examples:
    - /shaders/sample_wrap_mirror.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleMirror(tex: SAMPLER_TYPE, st: vec2f) -> vec4f {
    return SAMPLER_FNC( tex, mirror(st) ); 
}
