#include "linearDodge.glsl"
#include "linearBurn.glsl"

/*
author: Jamie Owen
description: Photoshop Linear Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendLinearLigth(<float|vec3> base, <float|vec3> blend [, <float> opacity])
licence: TODO
*/

#ifndef FNC_BLENDLINEARLIGHT
#define FNC_BLENDLINEARLIGHT
float blendLinearLigth(in float base, in float blend) {
  return blend < .5? blendLinearBurn(base, (2. * blend)): blendLinearDodge(base, (2. * (blend- .5)));
}

vec3 blendLinearLigth(in vec3 base, in vec3 blend) {
  return vec3(blendLinearLigth(base.r, blend.r),
              blendLinearLigth(base.g, blend.g),
              blendLinearLigth(base.b, blend.b));
}

vec3 blendLinearLigth(in vec3 base, in vec3 blend, in float opacity) {
	return (blendLinearLigth(base, blend) * opacity + base * (1. - opacity));
}
#endif
