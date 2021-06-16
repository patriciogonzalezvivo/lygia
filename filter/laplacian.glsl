/*
author: Patricio Gonzalez Vivo
description: laplacian filter
use: laplacian(<sampler2D> texture, <vec2> st, <vec2> pixels_scale [, <float> pixel_padding])
options:
    LAPLACIAN_TYPE: Return type, defaults to float
    LAPLACIAN_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,POS_UV).r
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef LAPLACIAN_TYPE
#define LAPLACIAN_TYPE vec3
#endif

#ifndef LAPLACIAN_SAMPLER_FNC
#define LAPLACIAN_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV).rgb
#endif

#ifndef LAPLACIAN_FNC
#define LAPLACIAN_FNC laplacian_w4
#endif

// #define LAPLACE_W4 0
// #define LAPLACE_W8 0
// #define LAPLACE_W12 0

// LAPLACE FILTER (highpass)
//                                  
//   0  1  0    1  1  1    1   2   1
//   1 -4  1    1 -8  1    2 -12   2
//   0  1  0    1  1  1    1   2   1
//    

#ifndef FNC_LAPLACIAN
#define FNC_LAPLACIAN

LAPLACIAN_TYPE laplacian_w4(sampler2D tex, vec2 st, vec2 pixel) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    acc += LAPLACIAN_SAMPLER_FNC(st) * 4.0;
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2(-1.0,  0.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 0.0, -1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 0.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0,  0.0) * pixel);
    return acc;
}

LAPLACIAN_TYPE laplacian_w8(sampler2D tex, vec2 st, vec2 pixel) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    acc += LAPLACIAN_SAMPLER_FNC(st) * 8.0;
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2(-1.0,  0.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 0.0, -1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 0.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0,  0.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2(-1.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0, -1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0,  1.0) * pixel);
    return acc;
}

LAPLACIAN_TYPE laplacian_w12(sampler2D tex, vec2 st, vec2 pixel) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    acc += LAPLACIAN_SAMPLER_FNC(st) * 12.0;
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2(-1.0,  0.0) * pixel) * 2.0;
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 0.0, -1.0) * pixel) * 2.0;
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 0.0,  1.0) * pixel) * 2.0;
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0,  0.0) * pixel) * 2.0;

    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2(-1.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0, -1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0,  1.0) * pixel);
    acc -= LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0,  1.0) * pixel);
    return acc;
}

LAPLACIAN_TYPE laplacian(sampler2D tex, vec2 st, vec2 pixel) {
    return LAPLACIAN_FNC(tex, st, pixel);
}

bool laplacian_isOutside(vec2 pos) {
    return (pos.x < 0.0 || pos.y < 0.0 || pos.x > 1.0 || pos.y > 1.0);
}

LAPLACIAN_TYPE laplacian_w4(sampler2D tex, vec2 st, vec2 pixel, float pixel_pad) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    vec2 uv = st * vec2(1.0 + pixel_pad * 2.0 * pixel) - pixel_pad * pixel;
    vec3 pixelShift = LAPLACIAN_TYPE(pixel, 0.0);

    if (!laplacian_isOutside(uv)) acc.xyz = 4.0 * LAPLACIAN_SAMPLER_FNC(uv);
    vec2 e = uv + pixelShift.xz;
    if (!laplacian_isOutside(e)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(e);
    vec2 n = uv + pixelShift.zy;
    if (!laplacian_isOutside(n)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(n);
    vec2 w = uv - pixelShift.xz;
    if (!laplacian_isOutside(w)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(w);
    vec2 s = uv - pixelShift.zy;
    if (!laplacian_isOutside(s)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(s);
    return acc;
}

LAPLACIAN_TYPE laplacian_w8(sampler2D tex, vec2 st, vec2 pixel, float pixel_pad) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    vec2 uv = st * vec2(1.0 + pixel_pad * 2.0 * pixel) - pixel_pad * pixel;
    vec3 pixelShift = vec3(pixel, 0.0);

    if (!laplacian_isOutside(uv)) acc.xyz = 8.0 * LAPLACIAN_SAMPLER_FNC(uv);
    vec2 e = uv + pixelShift.xz;
    if (!laplacian_isOutside(e)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(e);
    vec2 n = uv + pixelShift.zy;
    if (!laplacian_isOutside(n)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(n);
    vec2 w = uv - pixelShift.xz;
    if (!laplacian_isOutside(w)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(w);
    vec2 s = uv - pixelShift.zy;
    if (!laplacian_isOutside(s)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(s);

    vec2 ne = n + e;
    if (!laplacian_isOutside(e)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(e);
    vec2 nw = n + w;
    if (!laplacian_isOutside(n)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(n);
    vec2 se = s + e;
    if (!laplacian_isOutside(w)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(w);
    vec2 sw = s + w;
    if (!laplacian_isOutside(s)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(s);

    return acc;
}

LAPLACIAN_TYPE laplacian_w12(sampler2D tex, vec2 st, vec2 pixel, float pixel_pad) {
    LAPLACIAN_TYPE acc = LAPLACIAN_TYPE(0.0);
    vec2 uv = st * vec2(1.0 + pixel_pad * 2.0 * pixel) - pixel_pad * pixel;
    vec3 pixelShift = vec3(pixel, 0.0);

    if (!laplacian_isOutside(uv)) acc.xyz = 12.0 * LAPLACIAN_SAMPLER_FNC(uv);

    vec2 e = uv + pixelShift.xz;
    if (!laplacian_isOutside(e)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(e) * 2.0;
    vec2 n = uv + pixelShift.zy;
    if (!laplacian_isOutside(n)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(n) * 2.0;
    vec2 w = uv - pixelShift.xz;
    if (!laplacian_isOutside(w)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(w) * 2.0;
    vec2 s = uv - pixelShift.zy;
    if (!laplacian_isOutside(s)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(s) * 2.0;

    vec2 ne = n + e;
    if (!laplacian_isOutside(e)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(e);
    vec2 nw = n + w;
    if (!laplacian_isOutside(n)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(n);
    vec2 se = s + e;
    if (!laplacian_isOutside(w)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(w);
    vec2 sw = s + w;
    if (!laplacian_isOutside(s)) acc.xyz -= LAPLACIAN_SAMPLER_FNC(s);

    return acc;
}

LAPLACIAN_TYPE laplacian(sampler2D tex, vec2 st, vec2 pixel, float pixel_pad) {
    return LAPLACIAN_FNC(tex, st, pixel, pixel_pad);
}

#endif