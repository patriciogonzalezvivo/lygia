#include "../../math/absi.wgsl"
#include "../../sampler.wgsl"

/*
contributors: [Lingdong Huang, Patricio Gonzalez Vivo]
description: downscale for function for pyramids  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: <vec4> pyramidDownscale(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - PYRAMID_SAMPLE_FNC(TEX, UV): specific function for sampling the texture (texture2D(...) or texture(...))
    - PYRAMID_H1: |
        row/col weights for a 5x5 kernel convolution, and this numbers are using to multiply the result of the sampling based on their distance to the center
        flat average            0.2, 0.2, 0.2
        gaussian weights        0.4, 0.228, 0.056
        poisson fill            1.0334, 0.6836, 0.1507
        laplacian integration   0.7, 0.5, 0.15
license: MIT License (MIT C)opyright (c) 2020 Lingdong Huang
*/

// #define PYRAMID_H1 1.0334, 0.6836, 0.1507

// #define PYRAMID_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

// #define PYRAMID_DOWNSCALE_SAMPLE_FNC(TEX, UV) PYRAMID_SAMPLE_FNC(TEX, UV)
 
fn pyramidDownscale(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f) -> vec4f {
    let h1 = vec3f(PYRAMID_H1);

    let color = vec4f(0.0);
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            let uv = st + vec2f(float(dx), float(dy)) * pixel * 0.5;
            if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0)
                continue;
            color += PYRAMID_DOWNSCALE_SAMPLE_FNC(tex, uv) * h1[ absi(dx) ] * h1[ absi(dy) ];
        }
    }

    return color;
}
