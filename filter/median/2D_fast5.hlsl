#include "../../sampler.hlsl"

/*
contributors: [Morgan McGuire, Kyle Whitson]
description: |
    3x3 median filter, adapted from "A Fast, Small-Radius GPU Median Filter" 
    by Morgan McGuire in ShaderX6 https://casual-effects.com/research/McGuire2008Median/index.html
use: median2D_fast5(<SAMPLER_TYPE> texture, <float2> st, <float2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - MEDIAN2D_FAST5_TYPE: default float4
    - MEDIAN2D_FAST5_SAMPLER_FNC(TEX, UV): default texture2D(TEX, UV)
*/

#ifndef MEDIAN2D_FAST5_TYPE
#ifdef MEDIAN2D_TYPE
#define MEDIAN2D_FAST5_TYPE MEDIAN2D_TYPE
#else
#define MEDIAN2D_FAST5_TYPE float4
#endif
#endif

#ifndef MEDIAN2D_FAST5_SAMPLER_FNC
#ifdef MEDIAN_SAMPLER_FNC
#define MEDIAN2D_FAST5_SAMPLER_FNC(TEX, UV) MEDIAN_SAMPLER_FNC(TEX, UV)
#else
#define MEDIAN2D_FAST5_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef MEDIAN_S2
#define MEDIAN_S2(a, b) temp = a; a = min(a, b); b = max(temp, b);
#endif

#ifndef MEDIAN_2
#define MEDIAN_2(a, b) MEDIAN_S2(v[a], v[b]);
#endif

#ifndef FNC_MEDIAN2D_FAST5
#define FNC_MEDIAN2D_FAST5
#define MEDIAN_S2(a, b) temp = a; a = min(a, b); b = max(temp, b);
#define MEDIAN_2(a, b) MEDIAN_S2(v[a], v[b]);

#define MEDIAN_24(a, b, c, d, e, f, g, h) MEDIAN_2(a, b); MEDIAN_2(c, d); MEDIAN_2(e, f); MEDIAN_2(g, h);
#define MEDIAN_25(a, b, c, d, e, f, g, h, i, j) MEDIAN_24(a, b, c, d, e, f, g, h); MEDIAN_2(i, j);

MEDIAN2D_FAST5_TYPE median2D_fast5(in SAMPLER_TYPE tex, in float2 st, in float2 radius) {
    MEDIAN2D_FAST5_TYPE v[25];
    for (int dX = -2; dX <= 2; ++dX) {
        for (int dY = -2; dY <= 2; ++dY) {
            float2 offset = float2(float(dX), float(dY));
            // If a pixel in the window is located at (x+dX, y+dY), put it at index (dX + R)(2R + 1) + (dY + R) of the
            // pixel array. This will fill the pixel array, with the top left pixel of the window at pixel[0] and the
            // bottom right pixel of the window at pixel[N-1].
            v[(dX + 2) * 5 + (dY + 2)] = MEDIAN2D_FAST5_SAMPLER_FNC(tex, st + offset * radius);
        }
    }

    MEDIAN2D_FAST5_TYPE temp = MEDIAN2D_FAST5_TYPE(0.,0.,0.,0.);
    MEDIAN_25(0,  1,   3, 4,  2,  4,  2,  3,  6,  7);
    MEDIAN_25(5,  7,   5, 6,  9,  7,  1,  7,  1,  4);
    MEDIAN_25(12, 13, 11, 13, 11, 12, 15, 16, 14, 16);
    MEDIAN_25(14, 15, 18, 19, 17, 19, 17, 18, 21, 22);
    MEDIAN_25(20, 22, 20, 21, 23, 24, 2,  5,  3,  6);
    MEDIAN_25(0,  6,  0,  3,  4,  7,  1,  7,  1,  4);
    MEDIAN_25(11, 14, 8,  14, 8,  11, 12, 15, 9,  15);
    MEDIAN_25(9,  12, 13, 16, 10, 16, 10, 13, 20, 23);
    MEDIAN_25(17, 23, 17, 20, 21, 24, 18, 24, 18, 21);
    MEDIAN_25(19, 22, 8,  17, 9,  18, 0,  18, 0,  9);
    MEDIAN_25(10, 19, 1,  19, 1,  10, 11, 20, 2,  20);
    MEDIAN_25(2,  11, 12, 21, 3,  21, 3,  12, 13, 22);
    MEDIAN_25(4,  22, 4,  13, 14, 23, 5,  23, 5,  14);
    MEDIAN_25(15, 24, 6,  24, 6,  15, 7,  16, 7,  19);
    MEDIAN_25(3,  11, 5,  17, 11, 17, 9,  17, 4,  10);
    MEDIAN_25(6,  12, 7,  14, 4,  6,  4,  7,  12, 14);
    MEDIAN_25(10, 14, 6,  7,  10, 12, 6,  10, 6,  17);
    MEDIAN_25(12, 17, 7,  17, 7,  10, 12, 18, 7,  12);
    MEDIAN_24(10, 18, 12, 20, 10, 20, 10, 12);
    return v[12];
}
#endif
