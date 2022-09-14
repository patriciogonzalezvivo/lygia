/*
original_author: Jamie Owen
description: Photoshop Difference blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendDifference(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDDIFFERENCE
#define FNC_BLENDDIFFERENCE
float blendDifference(in float base, in float blend) {
    return abs(base-blend);
}

vec3 blendDifference(in vec3 base, in vec3 blend) {
    return abs(base-blend);
}

vec3 blendDifference(in vec3 base, in vec3 blend, in float opacity) {
    return (blendDifference(base, blend) * opacity + base * (1. - opacity));
}
#endif
