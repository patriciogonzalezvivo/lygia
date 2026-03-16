#include "../sample/clamp2edge.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Wrapper around three different edge detection algorithms Sobel, Prewitt, and directional Sobel
use: edge(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixels_scale)
options:
    - EDGE_FNC: Edge detection algorithm, defaults to edgeSobel, edgePrewitt & edgeSobel_directional also available
    - EDGE_TYPE: Return type, defaults to float
    - EDGE_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,TEX, UV).r
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
examples:
    - /shaders/filter_edge2D.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define EDGE_FNC edgeSobel

// #define EDGE_TYPE float

// #define EDGE_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV).r

#include "edge/prewitt.wgsl"
#include "edge/sobel.wgsl"
#include "edge/sobelDirectional.wgsl"

EDGE_TYPE edge(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    return EDGE_FNC(tex, st, offset);
}
