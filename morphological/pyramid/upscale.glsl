#include "../../math/absi.glsl"
#include "../../sampler.glsl"

/*
contributors:  [Lingdong Huang, Patricio Gonzalez Vivo]
description: upscale for function for pyramids  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: <vec4> pyramidUpscale(<SAMPLER_TYPE> tex0, SAMPLER_TYPE tex1, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - PYRAMID_UPSCALE0_SAMPLE_FNC(TEX, UV): sampling function for the previous level of the pyramid 
    - PYRAMID_UPSCALE1_SAMPLE_FNC(TEX, UV): sampling function for the same level of the pyramid in the downscale direction
    - PYRAMID_H1: simple average (0.2), for poisson fill (1.0334, 0.6836, 0.1507), for laplacian integration (0.7, 0.5, 0.15)
    - PYRAMID_H2: for poisson fill (0.0270), for laplacian integration (0.225)
    - PYRAMID_G: for poisson fill (0.7753, 0.0312), for laplacian integration (0.6, 0.25) (0.547 * 2.0, 0.175 * 2.0)
license: MIT License (MIT C)opyright (c) 2020 Lingdong Huang
*/

#ifndef PYRAMID_H1
#define PYRAMID_H1 1.0334, 0.6836, 0.1507
#endif

#ifndef PYRAMID_H2
#define PYRAMID_H2 0.0270
#endif

#ifndef PYRAMID_G
#define PYRAMID_G 0.7753, 0.0312
#endif

#ifndef PYRAMID_SAMPLE_FNC
#define PYRAMID_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef PYRAMID_UPSCALE0_SAMPLE_FNC
#define PYRAMID_UPSCALE0_SAMPLE_FNC(TEX, UV) PYRAMID_SAMPLE_FNC(TEX, UV)
#endif

#ifndef PYRAMID_UPSCALE1_SAMPLE_FNC
#define PYRAMID_UPSCALE1_SAMPLE_FNC(TEX, UV) PYRAMID_SAMPLE_FNC(TEX, UV)
#endif

#ifndef FNC_PYRAMID_UPSCALE
#define FNC_PYRAMID_UPSCALE
vec4 pyramidUpscale(SAMPLER_TYPE tex0, SAMPLER_TYPE tex1, vec2 st, vec2 pixel) {
    const vec3  h1 = vec3(PYRAMID_H1);
    const float h2 = PYRAMID_H2;
    const vec2  g  = vec2(PYRAMID_G);

    vec4 color = vec4(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel;
            if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0)
                continue;
                
            color += PYRAMID_UPSCALE0_SAMPLE_FNC(tex0, uv) * g[ absi(dx) ] * g[ absi(dy) ];
        }
    }

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel * 2.0;
            if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0)
                continue;

            color += PYRAMID_UPSCALE1_SAMPLE_FNC(tex1, uv) * h2 * h1[ absi(dx) ] * h1[ absi(dy) ];
        }
    }

    return color;
}
#endif