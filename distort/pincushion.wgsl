#include "../math/const.wgsl"
#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Pincushion distortion
use: barrel(SAMPLER_TYPE tex, <vec2> st [, <vec2|float> distance])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - PINCUSHION_TYPE: return type, defaults to vec3
    - PINCUSHION_SAMPLER_FNC: function used to sample the input texture, defaults to texture2D(TEX, UV).rgb
    - PINCUSHION_OCT_1: one octave of distortion
    - PINCUSHION_OCT_2: two octaves of distortion
    - PINCUSHION_OCT_3: three octaves of distortion
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/distort/barrel.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define PINCUSHION_TYPE vec3

// #define PINCUSHION_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb

fn pincushion(st: vec2f, pixel: vec2f, amt: f32) -> vec2f {
    float prop = pixel.x / pixel.y; // screen proroption
    vec2 m = vec2f(0.5, 0.5 / prop); // center coords
    vec2 d = st - m;                // vector from center to current fragment
    float dist = sqrt(dot(d, d));   // distance of pixel from center

    let power = (TAU / (2.0 * sqrt(dot(m, m)))) * -amt;
    let bind = (prop < 1.0)? m.x : m.y;
	
    let A = (power > 0.0)? tan(dist * power) : atan(dist * -power * 10.0);
    let B = (power > 0.0)? tan(bind * power) : atan(-power * bind * 10.0);

    let uv = m + normalize(d) * A * bind/B;
    return vec2f(uv.x, uv.y * prop);
}

PINCUSHION_TYPE pincushion(SAMPLER_TYPE tex, vec2 st, vec2 pixel, float amt) {
    let uv = pincushion(st, pixel, amt);
    return PINCUSHION_SAMPLER_FNC(tex, uv);
}
