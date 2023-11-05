#include "../../sample/clamp2edge.glsl"

/*
contributors: Brad Larson
description: Adapted version of directional Sobel edge detection from https://github.com/BradLarson/GPUImage2
use: edgeSobel_directional(<SAMPLER_TYPE> texture, <vec2> st, <vec2> pixels_scale)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - EDGESOBELDIRECTIONAL_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,TEX, UV).r
examples:
    - /shaders/filter_edge2D.frag
*/

#ifndef EDGESOBELDIRECTIONAL_SAMPLER_FNC
#ifdef EDGE_SAMPLER_FNC
#define EDGESOBELDIRECTIONAL_SAMPLER_FNC(TEX, UV) EDGE_SAMPLER_FNC(TEX, UV)
#else
#define EDGESOBELDIRECTIONAL_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV).r
#endif
#endif

#ifndef FNC_EDGESOBEL_DIRECTIONAL
#define FNC_EDGESOBEL_DIRECTIONAL
vec3 edgeSobelDirectional(in SAMPLER_TYPE tex, in vec2 st, in vec2 offset) {
    // get samples around pixel
    float tleft = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2(-offset.x, offset.y));
    float left = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2(-offset.x, 0.));
    float bleft = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2(-offset.x, -offset.y));
    float top = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2(0., offset.y));
    float bottom = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2(0., -offset.y));
    float tright = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + offset);
    float right = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2(offset.x, 0.));
    float bright = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2(offset.x, -offset.y));
    vec2 gradientDirection = vec2(0.);
    gradientDirection.x = -bleft - 2. * left - tleft + bright + 2. * right + tright;
    gradientDirection.y = -tleft - 2. * top - tright + bleft + 2. * bottom + bright;
    float gradientMagnitude = length(gradientDirection);
    vec2 normalizedDirection = normalize(gradientDirection);
    normalizedDirection = sign(normalizedDirection) * floor(abs(normalizedDirection) + .617316); // Offset by 1-sin(pi/8) to set to 0 if near axis, 1 if away
    normalizedDirection = (normalizedDirection + 1.) * .5; // Place -1.0 - 1.0 within 0 - 1.0
    return vec3(gradientMagnitude, normalizedDirection.x, normalizedDirection.y);
}
#endif
