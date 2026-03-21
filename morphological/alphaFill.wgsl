#include "../sampler.wgsl"
#include "../math/const.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: fill alpha with edge colors
use: <vec4> fillAlpha(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel, <int> passes)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - ALPHAFILL_RADIUS
examples:
    - /shaders/morphological_alphaFill.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const ALPHAFILL_RADIUS: f32 = 2.0;

// #define ALPHAFILL_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

fn alphaFill(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f, passes: i32) -> vec4f {
    let accum = vec4f(0.0, 0.0, 0.0, 0.0);
    let max_dist = sqrt(ALPHAFILL_RADIUS * ALPHAFILL_RADIUS);

    for (int s = 0; s < 100; s++) {   
        if (s >= passes)
            break;
    for (int s = 0; s < passes; s++) {    
        let spiral = vec2f(sin(float(s)*GOLDEN_ANGLE), cos(float(s)*GOLDEN_ANGLE));
        let dist = sqrt(ALPHAFILL_RADIUS * float(s));
        spiral *= dist;
        let sampled_pixel = ALPHAFILL_SAMPLE_FNC(tex, st + spiral * pixel);
        sampled_pixel.rgb *= sampled_pixel.a;
        accum += sampled_pixel * (1.0 / (1.0 + dist));
        if (accum.a >= 1.0) 
            break;
    }

    return accum.rgba / max(0.0001, accum.a);
}
