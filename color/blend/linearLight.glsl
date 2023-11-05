#include "linearDodge.glsl"
#include "linearBurn.glsl"

/*
contributors: Jamie Owen
description: Photoshop Linear Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendLinearLigth(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDLINEARLIGHT
#define FNC_BLENDLINEARLIGHT
float blendLinearLight(in float base, in float blend) {
  return blend < .5? blendLinearBurn(base, (2. * blend)): blendLinearDodge(base, (2. * (blend- .5)));
}

vec3 blendLinearLight(in vec3 base, in vec3 blend) {
  return vec3(blendLinearLight(base.r, blend.r),
              blendLinearLight(base.g, blend.g),
              blendLinearLight(base.b, blend.b));
}

vec3 blendLinearLight(in vec3 base, in vec3 blend, in float opacity) {
    return (blendLinearLight(base, blend) * opacity + base * (1. - opacity));
}
#endif
