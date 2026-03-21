#include "../../math/absi.wgsl"
#include "../../sampler.wgsl"

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

// #define PYRAMID_H1 1.0334, 0.6836, 0.1507

const PYRAMID_H2: f32 = 0.0270;

// #define PYRAMID_G 0.7753, 0.0312

// #define PYRAMID_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

// #define PYRAMID_UPSCALE0_SAMPLE_FNC(TEX, UV) PYRAMID_SAMPLE_FNC(TEX, UV)

// #define PYRAMID_UPSCALE1_SAMPLE_FNC(TEX, UV) PYRAMID_SAMPLE_FNC(TEX, UV)

fn pyramidUpscale(tex0: SAMPLER_TYPE, tex1: SAMPLER_TYPE, st: vec2f, pixel: vec2f) -> vec4f {
    let h1 = vec3f(PYRAMID_H1);
    let h2 = PYRAMID_H2;
    let g = vec2f(PYRAMID_G);

    let color = vec4f(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            let uv = st + vec2f(float(dx), float(dy)) * pixel;
            if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0)
                continue;
                
            color += PYRAMID_UPSCALE0_SAMPLE_FNC(tex0, uv) * g[ absi(dx) ] * g[ absi(dy) ];
        }
    }

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            let uv = st + vec2f(float(dx), float(dy)) * pixel * 2.0;
            if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0)
                continue;

            color += PYRAMID_UPSCALE1_SAMPLE_FNC(tex1, uv) * h2 * h1[ absi(dx) ] * h1[ absi(dy) ];
        }
    }

    return color;
}
