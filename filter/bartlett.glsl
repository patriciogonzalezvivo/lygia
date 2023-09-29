#include "../math/const.glsl"
#include "../math/saturate.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    The Bartlett filter is a low-pass filter, which means that it blurs the image. 
    It's a convolution that uses a piecewise linear kernel, approximating a "tent" or triangle.
    It's also known as bilinear or triangle filter.
    Based on https://www.shadertoy.com/view/3djSDt

use: bartlett(<SAMPLER_TYPE> texture, <vec2> st, <vec2> duv [, <int> kernelSize]])

options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BARTLETT_TYPE: default is vec3
    - BARTLETT_SAMPLER_FNC(TEX, UV): default texture2D(TEX, UV)
*/

#ifndef BARTLETT_TYPE
#define BARTLETT_TYPE vec4
#endif

#ifndef BARTLETT_SAMPLER_FNC
#define BARTLETT_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_BARTLETT
#define FNC_BARTLETT
BARTLETT_TYPE bartlett(in sampler2D tex, vec2 st, vec2 pixel, int radius) {
    vec2 coord = st / pixel;
    ivec2 pos = ivec2(floor(coord));
    float R = float(radius);
    BARTLETT_TYPE acc = BARTLETT_TYPE(0.0);
    float normalizationConstant = 3.0 / (PI * R*R);
    for(int y = pos.y - radius; y <= pos.y + radius;++y)
        for(int x = pos.x - radius; x <= pos.x + radius;++x) {
            vec2 xy = vec2(x,y);
            float mag = length(xy-coord);
            float weight = 1.0 - saturate(mag/R);
            acc += weight * BARTLETT_SAMPLER_FNC(tex, xy * pixel);
        }
    return normalizationConstant*acc;
}
#endif