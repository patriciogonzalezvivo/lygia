/*
original_author: Jamie Owen
description: Photoshop Darken blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendDarken(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDDARKEN
#define FNC_BLENDDARKEN
float blendDarken(in float base, in float blend) {
    return min(blend,base);
}

vec3 blendDarken(in vec3 base, in vec3 blend) {
    return vec3(blendDarken(base.r, blend.r),
                blendDarken(base.g, blend.g),
                blendDarken(base.b, blend.b));
}

vec3 blendDarken(in vec3 base, in vec3 blend, in float opacity) {
    return (blendDarken(base, blend) * opacity + base * (1. - opacity));
}
#endif
