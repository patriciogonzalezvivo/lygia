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

#ifndef FNC_LAPLACIAN
#define FNC_LAPLACIAN

LAPLACIAN_TYPE laplacian(sampler2D tex, vec2 st, vec2 pixel) {
    return    LAPLACIAN_SAMPLER_FNC(st) * 4.0
            - LAPLACIAN_SAMPLER_FNC(st + vec2(-1.0,  0.0) * pixel)
            - LAPLACIAN_SAMPLER_FNC(st + vec2( 0.0, -1.0) * pixel)
            - LAPLACIAN_SAMPLER_FNC(st + vec2( 0.0,  1.0) * pixel)
            - LAPLACIAN_SAMPLER_FNC(st + vec2( 1.0,  0.0) * pixel);
}

bool laplacian_isOutside(vec2 pos) {
    return (pos.x < 0.0 || pos.y < 0.0 || pos.x > 1.0 || pos.y > 1.0);
}

LAPLACIAN_TYPE laplacian(sampler2D tex, vec2 st, vec2 pixel, float pixel_pad) {
    LAPLACIAN_TYPE acum = LAPLACIAN_TYPE(0.0);
    vec2 uv = st * vec2(1.0 + pixel_pad * 2.0 * pixel) - pixel_pad * pixel;
    vec3 pixelShift = LAPLACIAN_TYPE(pixel, 0.0);

    if (!laplacian_isOutside(uv)) acum.xyz = 4.0 * LAPLACIAN_SAMPLER_FNC(uv).rgb;
    vec2 uv110 = uv + pixelShift.xz;
    if (!laplacian_isOutside(uv110)) acum.xyz -= LAPLACIAN_SAMPLER_FNC(uv110).rgb;
    vec2 uv101 = uv + pixelShift.zy;
    if (!laplacian_isOutside(uv101)) acum.xyz -= LAPLACIAN_SAMPLER_FNC(uv101).rgb;
    vec2 uv010 = uv - pixelShift.xz;
    if (!laplacian_isOutside(uv010)) acum.xyz -= LAPLACIAN_SAMPLER_FNC(uv010).rgb;
    vec2 uv001 = uv - pixelShift.zy;
    if (!laplacian_isOutside(uv001)) acum.xyz -= LAPLACIAN_SAMPLER_FNC(uv001).rgb;
    return acum;
}

#endif