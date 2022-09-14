/*
original_author: Jamie Owen
description: Photoshop Soft Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendSubtract(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDSUBTRACT
#define FNC_BLENDSUBTRACT
float blendSubtract(in float base, in float blend) {
    return max(base + blend - 1., 0.);
}

vec3 blendSubtract(in vec3 base, in vec3 blend) {
    return max(base + blend - vec3(1.), vec3(0.));
}

vec3 blendSubtract(in vec3 base, in vec3 blend, in float opacity) {
    return (blendSubtract(base, blend) * opacity + base * (1. - opacity));
}
#endif
