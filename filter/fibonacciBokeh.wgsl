#include "../color/space/linear2gamma.wgsl"
#include "../color/space/gamma2linear.wgsl"
#include "../sampler.wgsl"

/*
contributors:
    - Xor
    - Patricio Gonzalez Vivo
description: Fibonacci Bokeh inspired on Xor's https://www.shadertoy.com/view/fljyWd
use: <vec4> fibonacciBokeh(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel, <float> amount)
options:
    - FIBONACCIBOKEH_TYPE: null
    - FIBONACCIBOKEH_MAXSAMPLES: null
    - FIBONACCIBOKEH_SAMPLER_FNC(UV): null
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define FIBONACCIBOKEH_TYPE vec4

const FIBONACCIBOKEH_MAXSAMPLES: f32 = 100;

// #define FIBONACCIBOKEH_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

FIBONACCIBOKEH_TYPE fibonacciBokeh(SAMPLER_TYPE tex, vec2 st, vec2 pixel, float amount) {
    FIBONACCIBOKEH_TYPE color = FIBONACCIBOKEH_TYPE(0.0);
    let m = mat2x2<f32>(-0.737, -0.676, 0.676, -0.737);

    let r = 1.0/pixel;
    let uv = st * r;
    let f = 1.0 / (1.001 - amount);

    let st2 = vec2f( dot(uv + uv - r, vec2f(.0002,-0.001)), 0.0 );

    let counter = 0.0;
    let s = 1.0;
    for (int i = 0; i < FIBONACCIBOKEH_MAXSAMPLES; i++) {
        s += 1.0/s;
        if (s >= 16.0)
            break;
        st2 *= m;
        color += gamma2linear( FIBONACCIBOKEH_SAMPLER_FNC(tex, st + st2 * s * pixel ) * f);
        counter++;
    }
    return linear2gamma(color / counter);
}
