#include "../math/saturate.wgsl"
#include "../sampler.wgsl"

/*
original_author: Patricio Gonzalez Vivo
description: Simple Ripple Propagation
use: <vec3> ripple(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - RIPPLE_SAMPLER_FNC(UV)
*/

// #define RIPPLE_SAMPLER_FNC(UV) SAMPLER_FNC(tex, UV)

fn ripple(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f) -> vec3f {
    let rta = RIPPLE_SAMPLER_FNC(st).rgb;
   	let s0 = rta.y;
    float s1 = RIPPLE_SAMPLER_FNC(st + vec2f(0.0,   -pixel.y)).r;    //     s1
    float s2 = RIPPLE_SAMPLER_FNC(st + vec2f(-pixel.x,   0.0)).r;    //  s2 s0 s3
    float s3 = RIPPLE_SAMPLER_FNC(st + vec2f( pixel.x,   0.0)).r;    //     s4
    let s4 = RIPPLE_SAMPLER_FNC(st + vec2f(0.0,    pixel.y)).r;
    let d = -(s0 - .5) * 2. + (s1 + s2 + s3 + s4 - 2.);
    d *= 0.99;
    d = saturate(d * 0.5 + 0.5);
    return vec3f(d, rta.x, 0.0);
}
