#include "../../sampler.glsl"

/*
contributors: [Morgan McGuire, Kyle Whitson]
description: |
    3x3 median filter, adapted from "A Fast, Small-Radius GPU Median Filter" 
    by Morgan McGuire in ShaderX6 https://casual-effects.com/research/McGuire2008Median/index.html
use: median2D_fast3(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - MEDIAN2D_FAST3_TYPE: default vec4
    - MEDIAN2D_FAST3_SAMPLER_FNC(TEX, UV): default texture2D(TEX, UV)
examples:
    - /shaders/filter_median2D.frag
*/

#ifndef MEDIAN2D_FAST3_TYPE
#ifdef MEDIAN2D_TYPE
#define MEDIAN2D_FAST3_TYPE MEDIAN2D_TYPE
#else
#define MEDIAN2D_FAST3_TYPE vec4
#endif
#endif

#ifndef MEDIAN2D_FAST3_SAMPLER_FNC
#ifdef MEDIAN_SAMPLER_FNC
#define MEDIAN2D_FAST3_SAMPLER_FNC(TEX, UV) MEDIAN_SAMPLER_FNC(TEX, UV)
#else
#define MEDIAN2D_FAST3_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef MEDIAN_S2
#define MEDIAN_S2(a, b) temp = a; a = min(a, b); b = max(temp, b);
#endif

#ifndef MEDIAN_2
#define MEDIAN_2(a, b) MEDIAN_S2(v[a], v[b]);
#endif

#ifndef FNC_MEDIAN2D_FAST3
#define FNC_MEDIAN2D_FAST3
#define MEDIAN_MN3(a, b, c) MEDIAN_2(a, b); MEDIAN_2(a, c);
#define MEDIAN_MX3(a, b, c) MEDIAN_2(b, c); MEDIAN_2(a, c);
#define MEDIAN_MNMX3(a, b, c) MEDIAN_MX3(a, b, c); MEDIAN_2(a, b);                                                                // 3 exchanges
#define MEDIAN_MNMX4(a, b, c, d) MEDIAN_2(a, b); MEDIAN_2(c, d); MEDIAN_2(a, c); MEDIAN_2(b, d);                                  // 4 exchanges
#define MEDIAN_MNMX5(a, b, c, d, e) MEDIAN_2(a, b); MEDIAN_2(c, d); MEDIAN_MN3(a, c, e); MEDIAN_MX3(b, d, e);                     // 6 exchanges
#define MEDIAN_MNMX6(a, b, c, d, e, f) MEDIAN_2(a, d); MEDIAN_2(b, e); MEDIAN_2(c, f); MEDIAN_MN3(a, b, c); MEDIAN_MX3(d, e, f);  // 7 exchanges
MEDIAN2D_FAST3_TYPE median2D_fast3(in SAMPLER_TYPE tex, in vec2 st, in vec2 radius) {
    MEDIAN2D_FAST3_TYPE v[9];
    for (int dX = -1; dX <= 1; ++dX) {
        for (int dY = -1; dY <= 1; ++dY) {
            vec2 offset = vec2(float(dX), float(dY));
            // If a pixel in the window is located at (x+dX, y+dY), put it at index (dX + R)(2R + 1) + (dY + R) of the
            // pixel array. This will fill the pixel array, with the top left pixel of the window at pixel[0] and the
            // bottom right pixel of the window at pixel[N-1].
            v[(dX + 1) * 3 + (dY + 1)] = MEDIAN2D_FAST3_SAMPLER_FNC(tex, st + offset * radius);
        }
    }
    MEDIAN2D_FAST3_TYPE temp = MEDIAN2D_FAST3_TYPE(0.);
    MEDIAN_MNMX6(0, 1, 2, 3, 4, 5);
    MEDIAN_MNMX5(1, 2, 3, 4, 6);
    MEDIAN_MNMX4(2, 3, 4, 7);
    MEDIAN_MNMX3(3, 4, 8);
    return v[4];
}
#endif
