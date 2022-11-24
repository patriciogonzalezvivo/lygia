#include "../../math/absi.glsl"
#include "../../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: upscale for function for convolution pyramid  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: <vec4> POISSONFILLUpscale(<sampler2D> tex0, sampler2D tex1, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - POISSONFILL_H1: 1.0334, 0.6836, 0.1507
    - POISSONFILL_H2: 0.0270
    - POISSONFILL_G: 0.7753, 0.0312
*/

#ifndef POISSONFILL_H1
#define POISSONFILL_H1 1.0334, 0.6836, 0.1507
#endif

#ifndef POISSONFILL_H2
#define POISSONFILL_H2 0.0270
#endif

#ifndef POISSONFILL_G
#define POISSONFILL_G 0.7753, 0.0312
#endif

#ifndef FNC_POISSONFILL_UPSCALE
#define FNC_POISSONFILL_UPSCALE
vec4 poissonFillUpscale(sampler2D tex0, sampler2D tex1, vec2 st, vec2 pixel) {
    const vec3  h1 = vec3(POISSONFILL_H1);
    const float h2 = POISSONFILL_H2;
    const vec2  g  = vec2(POISSONFILL_G);

    vec4 color = vec4(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel;
            color += SAMPLER_FNC(tex0, uv) * g[ absi(dx) ] * g[ absi(dy) ];
        }
    }

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel * 2.;
            color += SAMPLER_FNC(tex1, uv) * h2 * h1[ absi(dx) ] * h1[ absi(dy) ];
        }
    }

    return color;
}
#endif