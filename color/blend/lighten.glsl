/*
original_author: Jamie Owen
description: Photoshop Lighten blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendLighten(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDLIGHTEN
#define FNC_BLENDLIGHTEN
float blendLighten(in float base, in float blend) {
    return max(blend, base);
}

vec3 blendLighten(in vec3 base, in vec3 blend) {
    return vec3(blendLighten(base.r, blend.r),
                blendLighten(base.g, blend.g),
                blendLighten(base.b, blend.b));
}

vec3 blendLighten(in vec3 base, in vec3 blend, in float opacity) {
    return (blendLighten(base, blend) * opacity + base * (1. - opacity));
}
#endif
