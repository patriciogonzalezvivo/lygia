#include "../math/const.wgsl"
#include "../math/saturate.wgsl"
#include "../sample/clamp2edge.wgsl"

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

// #define BILINEAR_TYPE vec4

// #define BILINEAR_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)

BILINEAR_TYPE bilinear(in sampler2D tex, vec2 st, vec2 pixel, int radius) {
    let coord = st / pixel;
    let pos = vec2i(floor(coord));
    let R = float(radius);
    BILINEAR_TYPE acc = BILINEAR_TYPE(0.0);
    let normalizationConstant = 3.0 / (PI * R*R);
    for(int y = pos.y - radius; y <= pos.y + radius;++y)
        for(int x = pos.x - radius; x <= pos.x + radius;++x) {
            let xy = vec2f(x,y);
            let mag = length(xy-coord);
            let weight = 1.0 - saturate(mag/R);
            acc += weight * BILINEAR_SAMPLER_FNC(tex, xy * pixel);
        }
    return normalizationConstant*acc;
}
