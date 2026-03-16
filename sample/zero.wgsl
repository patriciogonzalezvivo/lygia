#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: turns alpha to zero if it's outside the texture normalize coordinates
use: <vec4> sampleZero(<SAMPLER_TYPE> tex, <vec2> st);
options:
    - SAMPLER_FNC(TEX, UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleZero(tex: SAMPLER_TYPE, st: vec2f) -> vec4f {
    return SAMPLER_FNC( tex, st ) * vec4f(1.0,1.0,1.0, (st.x <= 0.0 || st.x >= 1.0 || st.y <= 0.0 || st.y >= 1.0)? 0.0 : 1.0); 
}

fn sampleZeroa(tex: SAMPLER_TYPE, st: vec2f, pad: f32) -> vec4f {
    return SAMPLER_FNC( tex, st ) * vec4f(1.0,1.0,1.0, (st.x <= pad || st.x >= 1.0-pad || st.y <= pad || st.y >= 1.0 - pad)? 0.0 : 1.0); 
}

fn sampleZerob(tex: SAMPLER_TYPE, st: vec2f, pad: vec2f) -> vec4f {
    return SAMPLER_FNC( tex, st ) * vec4f(1.0,1.0,1.0, (st.x <= pad.x || st.x >= 1.0-pad.x || st.y <= pad.y || st.y >= 1.0 - pad.y)? 0.0 : 1.0); 
}

fn sampleZeroc(tex: SAMPLER_TYPE, st: vec2f, pad: vec4f) -> vec4f {
    return SAMPLER_FNC( tex, st ) * vec4f(1.0,1.0,1.0, (st.x <= pad.x || st.x >= pad.z || st.y <= pad.y || st.y >= pad.w)? 0.0 : 1.0); 
}

fn sampleZerod(tex: SAMPLER_TYPE, st: vec2f, pad: AABB) -> vec4f {
    return SAMPLER_FNC( tex, st ) * vec4f(1.0,1.0,1.0, (st.x <= pad.min.x || st.x >= pad.max.x || st.y <= pad.min.y || st.y >= pad.max.y)? 0.0 : 1.0); 
}
