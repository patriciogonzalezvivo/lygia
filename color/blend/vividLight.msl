#include "colorBurn.glsl"
#include "colorDodge.glsl"

/*
contributors: Jamie Owen
description: Photoshop Vivid Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendVividLight(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDVIVIDLIGHT
#define FNC_BLENDVIVIDLIGHT
float blendVividLight(in float base, in float blend) {
    return (blend < .5)? blendColorBurn(base, (2.*blend)): blendColorDodge(base, (2. * (blend - .5)));
}

vec3 blendVividLight(in vec3 base, in vec3 blend) {
    return vec3(blendVividLight(base.r, blend.r),
                            blendVividLight(base.g, blend.g),
                            blendVividLight(base.b, blend.b));
}

vec3 blendVividLight(in vec3 base, in vec3 blend, in float opacity) {
    return (blendVividLight(base, blend) * opacity + base * (1. - opacity));
}
#endif
