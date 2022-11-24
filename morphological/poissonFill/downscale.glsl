#include "../../math/absi.glsl"
#include "../../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: downscale for function for convolution pyramid  https://www.cs.huji.ac.il/labs/cglab/projects/convpyr/data/convpyr-small.pdf
use: <vec4> convolutionPyramidDownscale(<sampler2D> tex, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - POISSONFILL_H1: 1.0334, 0.6836, 0.1507
*/

#ifndef POISSONFILL_H1
#define POISSONFILL_H1 1.0334, 0.6836, 0.1507
#endif
 
#ifndef FNC_POISSONFILL_DOWNSCALE
#define FNC_POISSONFILL_DOWNSCALE
vec4 poissonFillDownscale(sampler2D tex, vec2 st, vec2 pixel) {
    const vec3 h1 = vec3(POISSONFILL_H1);

    vec4 color = vec4(0.0);
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 uv = st + vec2(float(dx), float(dy)) * pixel;
            if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0)
                continue;
            color += SAMPLER_FNC(tex, uv) * h1[ absi(dx) ] * h1[ absi(dy) ];
        }
    }

    return color;
}
#endif