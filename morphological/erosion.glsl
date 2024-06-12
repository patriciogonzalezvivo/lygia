#include "../sampler.glsl"
#include "../math/saturate.glsl"
#include "../math/sum.glsl"

/*
contributors: nan
description: |
    morphological erosion operation. Based on: 
        https://lettier.github.io/3d-game-shaders-for-beginners/dilation.html
        https://www.shadertoy.com/view/WsyXWc
use: erode(<SAMPLER_TYPE> texture, <float2> st, <float2> pixels_scale, <int> passes)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - EROSION_TYPE
    - EROSION_SAMPLE_FNC(TEX, UV)
*/

#ifndef EROSION_TYPE
#define EROSION_TYPE float
#endif

#ifndef EROSION_SAMPLE_FNC
#define EROSION_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).x
#endif

#ifndef FNC_EROSION
#define FNC_EROSION

EROSION_TYPE erosion(SAMPLER_TYPE tex, vec2 st, vec2 pixel, int radius) {

    float invKR = 1.0 / float(radius);
    EROSION_TYPE acc = EROSION_TYPE(1.0);
    float w = 0.0;
    for(int i = -radius; i <= radius; ++i)
    for(int j = -radius; j <= radius; ++j) {
        vec2 rxy = vec2(ivec2(i, j));
        vec2 kst = rxy * invKR;
        vec2 texOffset = st + rxy * pixel;
        float kernel = saturate(1.0 - dot(kst, kst));
        EROSION_TYPE t = EROSION_SAMPLE_FNC(tex, texOffset);
        EROSION_TYPE v = t - kernel;
        if (sum(v) < sum(acc)) {
            acc = v;
            w = kernel;
        }
    }
    
    return acc + w;
}

#endif