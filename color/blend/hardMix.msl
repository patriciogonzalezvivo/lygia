#include "vividLight.glsl"

/*
contributors: Jamie Owen
description: Photoshop Hard Mix blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendHardMix(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDHARDMIX
#define FNC_BLENDHARDMIX
float blendHardMix(in float base, in float blend) {
    return (blendVividLight(base, blend) < .5)? 0.: 1.;
}

vec3 blendHardMix(in vec3 base, in vec3 blend) {
    return vec3(blendHardMix(base.r, blend.r),
                blendHardMix(base.g, blend.g),
                blendHardMix(base.b, blend.b));
}

vec3 blendHardMix(in vec3 base, in vec3 blend, in float opacity) {
    return (blendHardMix(base, blend) * opacity + base * (1. - opacity));
}
#endif
