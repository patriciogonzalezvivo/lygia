/*
original_author: Patricio Gonzalez Vivo
description: simple two dimentional box blur, so can be apply in a single pass
use: boxBlur1D_fast9(<sampler2D> texture, <vec2> st, <vec2> pixel_direction)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BOXBLUR2D_FAST9_TYPE: Default is `vec4`
    - BOXBLUR2D_FAST9_SAMPLER_FNC(POS_UV): Default is `texture2D(tex, POS_UV)`
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef BOXBLUR2D_FAST9_TYPE
#ifdef BOXBLUR_TYPE
#define BOXBLUR2D_FAST9_TYPE BOXBLUR_TYPE
#else
#define BOXBLUR2D_FAST9_TYPE vec4
#endif
#endif

#ifndef BOXBLUR2D_FAST9_SAMPLER_FNC
#ifdef BOXBLUR_SAMPLER_FNC
#define BOXBLUR2D_FAST9_SAMPLER_FNC(POS_UV) BOXBLUR_SAMPLER_FNC(POS_UV)
#else
#define BOXBLUR2D_FAST9_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif
#endif

#ifndef FNC_BOXBLUR2D_FAST9
#define FNC_BOXBLUR2D_FAST9
BOXBLUR2D_FAST9_TYPE boxBlur2D_fast9(in sampler2D tex, in vec2 st, in vec2 offset) {
    BOXBLUR2D_FAST9_TYPE color = BOXBLUR2D_FAST9_SAMPLER_FNC(st);           // center
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(-offset.x, offset.y));  // tleft
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(-offset.x, 0.));        // left
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(-offset.x, -offset.y)); // bleft
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(0., offset.y));         // top
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(0., -offset.y));        // bottom
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + offset);                     // tright
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(offset.x, 0.));         // right
    color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(offset.x, -offset.y));  // bright
    return color * 0.1111111111; // 1./9.
}
#endif
