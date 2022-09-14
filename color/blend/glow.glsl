#include "reflect.glsl"

/*
original_author: Jamie Owen
description: Photoshop Glow blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendGlow(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDGLOW
#define FNC_BLENDGLOW
float blendGlow(in float base, in float blend) {
    return blendReflect(blend, base);
}

vec3 blendGlow(in vec3 base, in vec3 blend) {
    return blendReflect(blend, base);
}

vec3 blendGlow(in vec3 base, in vec3 blend, in float opacity) {
    return (blendGlow(base, blend) * opacity + base * (1. - opacity));
}
#endif
