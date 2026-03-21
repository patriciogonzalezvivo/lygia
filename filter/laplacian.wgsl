#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Laplacian filter
use: laplacian(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixels_scale [, <float> pixel_padding])
options:
    - LAPLACIAN_TYPE: Return type, defaults to float
    - LAPLACIAN_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,TEX, UV).r
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
examples:
    - /shaders/filter_laplacian2D.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define LAPLACIAN_TYPE vec4

// #define LAPLACIAN_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

// #define LAPLACIAN_FNC laplacian_w4

// #define LAPLACE_W4 0
// #define LAPLACE_W8 0
// #define LAPLACE_W12 0

// LAPLACE FILTER (highpass)
//                                  
//   0  1  0    1  1  1    1   2   1
//   1 -4  1    1 -8  1    2 -12   2
//   0  1  0    1  1  1    1   2   1
//    

LAPLACIAN_TYPE laplacian_w4(SAMPLER_TYPE tex, vec2 st, vec2 pixel) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    acc += LAPLACIAN_SAMPLER_FNC(tex, st) * 4.0;
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f(-1.0,  0.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 0.0, -1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 0.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 1.0,  0.0) * pixel);
    return acc;
}

LAPLACIAN_TYPE laplacian_w8(SAMPLER_TYPE tex, vec2 st, vec2 pixel) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    acc += LAPLACIAN_SAMPLER_FNC(tex, st) * 8.0;
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f(-1.0,  0.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 0.0, -1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 0.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 1.0,  0.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f(-1.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 1.0, -1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 1.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 1.0,  1.0) * pixel);
    return acc;
}

LAPLACIAN_TYPE laplacian_w12(SAMPLER_TYPE tex, vec2 st, vec2 pixel) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    acc += LAPLACIAN_SAMPLER_FNC(tex, st) * 12.0;
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f(-1.0,  0.0) * pixel) * 2.0;
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 0.0, -1.0) * pixel) * 2.0;
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 0.0,  1.0) * pixel) * 2.0;
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 1.0,  0.0) * pixel) * 2.0;

    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f(-1.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 1.0, -1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 1.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(tex, st + vec2f( 1.0,  1.0) * pixel);
    return acc;
}

LAPLACIAN_TYPE laplacian(SAMPLER_TYPE tex, vec2 st, vec2 pixel) {
    return LAPLACIAN_FNC(tex, st, pixel);
}

fn laplacian_isOutside(pos: vec2f) -> bool {
    return (pos.x < 0.0 || pos.y < 0.0 || pos.x > 1.0 || pos.y > 1.0);
}

LAPLACIAN_TYPE laplacian_w4(SAMPLER_TYPE tex, vec2 st, vec2 pixel, float pixel_pad) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    let uv = st * vec2f(1.0 + pixel_pad * 2.0 * pixel) - pixel_pad * pixel;
    let pixelShift = vec3f(pixel, 0.0);

    if (!laplacian_isOutside(uv)) acc = 4.0 * LAPLACIAN_SAMPLER_FNC(tex, uv);
    let e = uv + pixelShift.xz;
    if (!laplacian_isOutside(e)) acc -= LAPLACIAN_SAMPLER_FNC(tex, e);
    let n = uv + pixelShift.zy;
    if (!laplacian_isOutside(n)) acc -= LAPLACIAN_SAMPLER_FNC(tex, n);
    let w = uv - pixelShift.xz;
    if (!laplacian_isOutside(w)) acc -= LAPLACIAN_SAMPLER_FNC(tex, w);
    let s = uv - pixelShift.zy;
    if (!laplacian_isOutside(s)) acc -= LAPLACIAN_SAMPLER_FNC(tex, s);
    return acc;
}

LAPLACIAN_TYPE laplacian_w8(SAMPLER_TYPE tex, vec2 st, vec2 pixel, float pixel_pad) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    let uv = st * vec2f(1.0 + pixel_pad * 2.0 * pixel) - pixel_pad * pixel;
    let pixelShift = vec3f(pixel, 0.0);

    if (!laplacian_isOutside(uv)) acc = 8.0 * LAPLACIAN_SAMPLER_FNC(tex, uv);
    let e = uv + pixelShift.xz;
    if (!laplacian_isOutside(e)) acc -= LAPLACIAN_SAMPLER_FNC(tex, e);
    let n = uv + pixelShift.zy;
    if (!laplacian_isOutside(n)) acc -= LAPLACIAN_SAMPLER_FNC(tex, n);
    let w = uv - pixelShift.xz;
    if (!laplacian_isOutside(w)) acc -= LAPLACIAN_SAMPLER_FNC(tex, w);
    let s = uv - pixelShift.zy;
    if (!laplacian_isOutside(s)) acc -= LAPLACIAN_SAMPLER_FNC(tex, s);

    let ne = n + e;
    if (!laplacian_isOutside(e)) acc -= LAPLACIAN_SAMPLER_FNC(tex, e);
    let nw = n + w;
    if (!laplacian_isOutside(n)) acc -= LAPLACIAN_SAMPLER_FNC(tex, n);
    let se = s + e;
    if (!laplacian_isOutside(w)) acc -= LAPLACIAN_SAMPLER_FNC(tex, w);
    let sw = s + w;
    if (!laplacian_isOutside(s)) acc -= LAPLACIAN_SAMPLER_FNC(tex, s);

    return acc;
}

LAPLACIAN_TYPE laplacian_w12(SAMPLER_TYPE tex, vec2 st, vec2 pixel, float pixel_pad) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    let uv = st * vec2f(1.0 + pixel_pad * 2.0 * pixel) - pixel_pad * pixel;
    let pixelShift = vec3f(pixel, 0.0);

    if (!laplacian_isOutside(uv)) acc = 12.0 * LAPLACIAN_SAMPLER_FNC(tex, uv);

    let e = uv + pixelShift.xz;
    if (!laplacian_isOutside(e)) acc -= LAPLACIAN_SAMPLER_FNC(tex, e) * 2.0;
    let n = uv + pixelShift.zy;
    if (!laplacian_isOutside(n)) acc -= LAPLACIAN_SAMPLER_FNC(tex, n) * 2.0;
    let w = uv - pixelShift.xz;
    if (!laplacian_isOutside(w)) acc -= LAPLACIAN_SAMPLER_FNC(tex, w) * 2.0;
    let s = uv - pixelShift.zy;
    if (!laplacian_isOutside(s)) acc -= LAPLACIAN_SAMPLER_FNC(tex, s) * 2.0;

    let ne = n + e;
    if (!laplacian_isOutside(e)) acc -= LAPLACIAN_SAMPLER_FNC(tex, e);
    let nw = n + w;
    if (!laplacian_isOutside(n)) acc -= LAPLACIAN_SAMPLER_FNC(tex, n);
    let se = s + e;
    if (!laplacian_isOutside(w)) acc -= LAPLACIAN_SAMPLER_FNC(tex, w);
    let sw = s + w;
    if (!laplacian_isOutside(s)) acc -= LAPLACIAN_SAMPLER_FNC(tex, s);

    return acc;
}

LAPLACIAN_TYPE laplacian(SAMPLER_TYPE tex, vec2 st, vec2 pixel, float pixel_pad) {
    return LAPLACIAN_FNC(tex, st, pixel, pixel_pad);
}
