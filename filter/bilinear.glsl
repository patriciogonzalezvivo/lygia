#include "../math/const.glsl"
#include "../math/saturate.glsl"
#include "../sample/clamp2edge.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: "Bilinear or Bartlett filter, a low-pass filter, which means that it\
    \ blurs the image. \nIt's a convolution that uses a piecewise linear kernel, approximating\
    \ a \"tent\" or triangle.\nBased on https://www.shadertoy.com/view/3djSDt\n"
use: bilinear(<SAMPLER_TYPE> texture, <vec2> st, <vec2> duv [, <int> kernelSize]])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BILINEAR_TYPE: default is vec3
    - BILINEAR_SAMPLER_FNC(TEX, UV): default texture2D(TEX, UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef BILINEAR_TYPE
#define BILINEAR_TYPE vec4
#endif

#ifndef BILINEAR_SAMPLER_FNC
#define BILINEAR_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)
#endif

#ifndef FNC_BILINEAR
#define FNC_BILINEAR
BILINEAR_TYPE bilinear(in sampler2D tex, vec2 st, vec2 pixel, int radius) {
    vec2 coord = st / pixel;
    ivec2 pos = ivec2(floor(coord));
    float R = float(radius);
    BILINEAR_TYPE acc = BILINEAR_TYPE(0.0);
    float normalizationConstant = 3.0 / (PI * R*R);
    for(int y = pos.y - radius; y <= pos.y + radius;++y)
        for(int x = pos.x - radius; x <= pos.x + radius;++x) {
            vec2 xy = vec2(x,y);
            float mag = length(xy-coord);
            float weight = 1.0 - saturate(mag/R);
            acc += weight * BILINEAR_SAMPLER_FNC(tex, xy * pixel);
        }
    return normalizationConstant*acc;
}
#endif