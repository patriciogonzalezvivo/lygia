#include "../color/space/linear2gamma.glsl"
#include "../color/space/gamma2linear.glsl"
#include "../sampler.glsl"

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

#ifndef FIBONACCIBOKEH_TYPE
#define FIBONACCIBOKEH_TYPE vec4
#endif

#ifndef FIBONACCIBOKEH_MAXSAMPLES
#define FIBONACCIBOKEH_MAXSAMPLES 100
#endif

#ifndef FIBONACCIBOKEH_SAMPLER_FNC
#define FIBONACCIBOKEH_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_FIBONACCIBOKEH
#define FNC_FIBONACCIBOKEH

FIBONACCIBOKEH_TYPE fibonacciBokeh(SAMPLER_TYPE tex, vec2 st, vec2 pixel, float amount) {
    FIBONACCIBOKEH_TYPE color = FIBONACCIBOKEH_TYPE(0.0);
    const mat2 m = mat2(-0.737, -0.676, 0.676, -0.737) ;

    vec2 r = 1.0/pixel;
    vec2 uv = st * r;
    float f = 1.0 / (1.001 - amount);

    vec2 st2 = vec2( dot(uv + uv - r, vec2(.0002,-0.001)), 0.0 );

    float counter = 0.0;
    float s = 1.0;
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

#endif
