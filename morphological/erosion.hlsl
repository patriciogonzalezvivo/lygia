#include "../sampler.hlsl"
#include "../math/sum.hlsl"

/*
contributors: nan
description: |
    morphological erosion operation. Based on: 
        https://lettier.github.io/3d-game-shaders-for-beginners/dilation.html
        https://www.shadertoy.com/view/WsyXWc
use: erode(<SAMPLER_TYPE> texture, <float2> st, <float2> pixels_scale, <int> passes)
options:
    - SAMPLER_FNC(TEX, UV)
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

EROSION_TYPE erosion(SAMPLER_TYPE tex,float2 st,float2 pixel, int radius) {

    float invKR = 1.0 / float(radius);
    EROSION_TYPE acc = EROSION_TYPE(1.0);
    float w = 0.0;
    for(int i = -radius; i <= radius; ++i)
    for(int j = -radius; j <= radius; ++j) {
        float2 rxy =float2(float(i), float(j));
        float2 kst = rxy * invKR;
        float2 texOffset = st + rxy * pixel;
        float kernel = saturate(1.0 - dot(kst, kst));
        EROSION_TYPE tex = EROSION_SAMPLE_FNC(tex, texOffset);
        EROSION_TYPE v = tex - kernel;
        if (sum(v) < sum(acc)) {
            acc = v;
            w = kernel;
        }
    }
    
    return acc + w;
}

#endif