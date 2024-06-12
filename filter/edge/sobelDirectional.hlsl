#include "../../sampler.hlsl"

/*
contributors: Brad Larson
description: Adapted version of directional Sobel edge detection from https://github.com/BradLarson/GPUImage2
use: edgeSobel_directional(<SAMPLER_TYPE> texture, <float2> st, <float2> pixels_scale)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - EDGESOBELDIRECTIONAL_SAMPLER_FNC: Function used to sample the input texture, defaults to texture2D(tex,TEX, UV).r
*/

#ifndef EDGESOBELDIRECTIONAL_SAMPLER_FNC
#ifdef EDGE_SAMPLER_FNC
#define EDGESOBELDIRECTIONAL_SAMPLER_FNC(TEX, UV) EDGE_SAMPLER_FNC(TEX, UV)
#else
#define EDGESOBELDIRECTIONAL_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r
#endif
#endif

#ifndef FNC_EDGESOBEL_DIRECTIONAL
#define FNC_EDGESOBEL_DIRECTIONAL
float3 edgeSobelDirectional(in SAMPLER_TYPE tex, in float2 st, in float2 offset) {
    // get samples around pixel
    float tleft = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + float2(-offset.x, offset.y));
    float left = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + float2(-offset.x, 0.));
    float bleft = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + float2(-offset.x, -offset.y));
    float top = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + float2(0., offset.y));
    float bottom = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + float2(0., -offset.y));
    float tright = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + offset);
    float right = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + float2(offset.x, 0.));
    float bright = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + float2(offset.x, -offset.y));
    float2 gradientDirection = float2(0., 0.);
    gradientDirection.x = -bleft - 2. * left - tleft + bright + 2. * right + tright;
    gradientDirection.y = -tleft - 2. * top - tright + bleft + 2. * bottom + bright;
    float gradientMagnitude = length(gradientDirection);
    float2 normalizedDirection = normalize(gradientDirection);
    normalizedDirection = sign(normalizedDirection) * floor(abs(normalizedDirection) + .617316); // Offset by 1-sin(pi/8) to set to 0 if near axis, 1 if away
    normalizedDirection = (normalizedDirection + 1.) * .5; // Place -1.0 - 1.0 within 0 - 1.0
    return float3(gradientMagnitude, normalizedDirection.x, normalizedDirection.y);
}
#endif
