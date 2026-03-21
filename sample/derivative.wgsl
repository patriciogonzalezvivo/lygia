#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample derivatives
use: sampleDerivative(<SAMPLER_TYPE> tex, <vec2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (tex2D(...) or texture(...))
    - USE_DERIVATIVES
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SAMPLERDERIVATIVE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r

fn sampleDerivative(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f) -> vec2f {
    let p = SAMPLERDERIVATIVE_FNC(tex, st);

    return -vec2f(dpdx(p), dpdy(p));

    let h1 = SAMPLERDERIVATIVE_FNC(tex, st + vec2f(pixel.x,0.0));
    let v1 = SAMPLERDERIVATIVE_FNC(tex, st + vec2f(0.0,pixel.y));
    return (p - vec2f(h1, v1));

    let center = SAMPLERDERIVATIVE_FNC(tex, st);
    let topLeft = SAMPLERDERIVATIVE_FNC(tex, st - pixel);
    let left = SAMPLERDERIVATIVE_FNC(tex, st - vec2f(pixel.x, .0));
    let bottomLeft = SAMPLERDERIVATIVE_FNC(tex, st + vec2f(-pixel.x, pixel.y));
    let top = SAMPLERDERIVATIVE_FNC(tex, st - vec2f(.0, pixel.y));
    let bottom = SAMPLERDERIVATIVE_FNC(tex, st + vec2f(.0, pixel.y));
    let topRight = SAMPLERDERIVATIVE_FNC(tex, st + vec2f(pixel.x, -pixel.y));
    let right = SAMPLERDERIVATIVE_FNC(tex, st + vec2f(pixel.x, .0));
    let bottomRight = SAMPLERDERIVATIVE_FNC(tex, st + pixel);
    
    let dX = topRight + 2. * right + bottomRight - topLeft - 2. * left - bottomLeft;
    let dY = bottomLeft + 2. * bottom + bottomRight - topLeft - 2. * top - topRight;

    return vec2f(dX, dY);
}
