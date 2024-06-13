#include "../sampler.glsl"
#include "../math/const.glsl"

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

#ifndef ALPHAFILL_RADIUS
#define ALPHAFILL_RADIUS 2.0
#endif

#ifndef ALPHAFILL_SAMPLE_FNC
#define ALPHAFILL_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_ALPHAFILL
#define FNC_ALPHAFILL
vec4 alphaFill(SAMPLER_TYPE tex, vec2 st, vec2 pixel, int passes) {
    vec4 accum = vec4(0.0, 0.0, 0.0, 0.0);
    float max_dist = sqrt(ALPHAFILL_RADIUS * ALPHAFILL_RADIUS);

    #if defined(PLATFORM_WEBGL)
    for (int s = 0; s < 100; s++) {   
        if (s >= passes)
            break;
    #else 
    for (int s = 0; s < passes; s++) {    
    #endif
        vec2 spiral = vec2(sin(float(s)*GOLDEN_ANGLE), cos(float(s)*GOLDEN_ANGLE));
        float dist = sqrt(ALPHAFILL_RADIUS * float(s));
        spiral *= dist;
        vec4 sampled_pixel = ALPHAFILL_SAMPLE_FNC(tex, st + spiral * pixel);
        sampled_pixel.rgb *= sampled_pixel.a;
        accum += sampled_pixel * (1.0 / (1.0 + dist));
        if (accum.a >= 1.0) 
            break;
    }

    return accum.rgba / max(0.0001, accum.a);
}
#endif