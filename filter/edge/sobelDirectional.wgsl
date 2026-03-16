#include "../../sample/clamp2edge.wgsl"

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

// #define EDGESOBELDIRECTIONAL_SAMPLER_FNC(TEX, UV) EDGE_SAMPLER_FNC(TEX, UV)
// #define EDGESOBELDIRECTIONAL_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV).r

fn edgeSobelDirectional(tex: SAMPLER_TYPE, st: vec2f, offset: vec2f) -> vec3f {
    // get samples around pixel
    let tleft = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2f(-offset.x, offset.y));
    let left = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2f(-offset.x, 0.));
    let bleft = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2f(-offset.x, -offset.y));
    let top = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2f(0., offset.y));
    let bottom = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2f(0., -offset.y));
    let tright = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + offset);
    let right = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2f(offset.x, 0.));
    let bright = EDGESOBELDIRECTIONAL_SAMPLER_FNC(tex, st + vec2f(offset.x, -offset.y));
    let gradientDirection = vec2f(0.);
    gradientDirection.x = -bleft - 2. * left - tleft + bright + 2. * right + tright;
    gradientDirection.y = -tleft - 2. * top - tright + bleft + 2. * bottom + bright;
    let gradientMagnitude = length(gradientDirection);
    let normalizedDirection = normalize(gradientDirection);
    normalizedDirection = sign(normalizedDirection) * floor(abs(normalizedDirection) + .617316); // Offset by 1-sin(pi/8) to set to 0 if near axis, 1 if away
    normalizedDirection = (normalizedDirection + 1.) * .5; // Place -1.0 - 1.0 within 0 - 1.0
    return vec3f(gradientMagnitude, normalizedDirection.x, normalizedDirection.y);
}
