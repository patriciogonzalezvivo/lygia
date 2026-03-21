#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: bicubic filter sampling
use: <vec4> sampleBicubic(<SAMPLER_TYPE> tex, <vec2> st, <vec2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
examples:
    - /shaders/sample_filter_bicubic.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleBicubic(v: f32) -> vec4f {
    let n = vec4f(1.0, 2.0, 3.0, 4.0) - v;
    let s = n * n * n;
    var o: vec4f;
    o.x = s.x;
    o.y = s.y - 4.0 * s.x;
    o.z = s.z - 4.0 * s.y + 6.0 * s.x;
    o.w = 6.0 - o.x - o.y - o.z;
    return o;
}

fn sampleBicubica(tex: SAMPLER_TYPE, st: vec2f, texResolution: vec2f) -> vec4f {
    let pixel = 1.0 / texResolution;
    st = st * texResolution - 0.5;

    let fxy = fract(st);
    st -= fxy;

    let xcubic = sampleBicubic(fxy.x);
    let ycubic = sampleBicubic(fxy.y);

    let c = st.xxyy + vec2 (-0.5, 1.5).xyxy;

    let s = vec4f(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    let offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= pixel.xxyy;

    let sample0 = SAMPLER_FNC(tex, offset.xz);
    let sample1 = SAMPLER_FNC(tex, offset.yz);
    let sample2 = SAMPLER_FNC(tex, offset.xw);
    let sample3 = SAMPLER_FNC(tex, offset.yw);

    let sx = s.x / (s.x + s.y);
    let sy = s.z / (s.z + s.w);

    return mix( mix(sample3, sample2, sx), 
                mix(sample1, sample0, sx), 
                sy);
}
