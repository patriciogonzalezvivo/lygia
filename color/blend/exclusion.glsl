/*
original_author: Jamie Owen
description: Photoshop Exclusion blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendExclusion(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDEXCLUSION
#define FNC_BLENDEXCLUSION
float blendExclusion(in float base, in float blend) {
    return base + blend - 2. * base * blend;
}

vec3 blendExclusion(in vec3 base, in vec3 blend) {
    return base + blend - 2. * base * blend;
}

vec3 blendExclusion(in vec3 base, in vec3 blend, in float opacity) {
    return (blendExclusion(base, blend) * opacity + base * (1. - opacity));
}
#endif
