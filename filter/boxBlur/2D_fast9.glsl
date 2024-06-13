#include "../../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple two dimentional box blur, so can be apply in a single pass
use: boxBlur1D_fast9(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel_direction)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BOXBLUR2D_FAST9_TYPE: Default is `vec4`
    - BOXBLUR2D_FAST9_SAMPLER_FNC(TEX, UV): Default is `texture2D(tex, TEX, UV)`
examples:
    - /shaders/filter_boxBlur2D.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef BOXBLUR2D_FAST9_TYPE
#ifdef BOXBLUR_TYPE
#define BOXBLUR2D_FAST9_TYPE BOXBLUR_TYPE
#else
#define BOXBLUR2D_FAST9_TYPE vec4
#endif
#endif

#ifndef BOXBLUR2D_FAST9_SAMPLER_FNC
#ifdef BOXBLUR_SAMPLER_FNC
#define BOXBLUR2D_FAST9_SAMPLER_FNC(TEX, UV) BOXBLUR_SAMPLER_FNC(TEX, UV)
#else
#define BOXBLUR2D_FAST9_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef FNC_BOXBLUR2D_FAST9
#define FNC_BOXBLUR2D_FAST9
BOXBLUR2D_FAST9_TYPE boxBlur2D_fast9(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    BOXBLUR2D_FAST9_TYPE color = BOXBLUR2D_FAST9_SAMPLER_FNC(tex, st);          // center
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(tex, st + vec2(-offset.x, offset.y));  // tleft
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(tex, st + vec2(-offset.x, 0.));        // left
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(tex, st + vec2(-offset.x, -offset.y)); // bleft
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(tex, st + vec2(0., offset.y));         // top
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(tex, st + vec2(0., -offset.y));        // bottom
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(tex, st + offset);                     // tright
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(tex, st + vec2(offset.x, 0.));         // right
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(tex, st + vec2(offset.x, -offset.y));  // bright
    return color * 0.1111111111; // 1./9.
}
#endif
