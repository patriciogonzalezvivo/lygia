/*
original_author: Jamie Owen
description: Photoshop Linear Burn blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendLinearBurn(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDLINEARBURN
#define FNC_BLENDLINEARBURN
float blendLinearBurn(in float base, in float blend) {
  // Note : Same implementation as BlendSubtractf
    return max(base + blend - 1., 0.);
}

vec3 blendLinearBurn(in vec3 base, in vec3 blend) {
  // Note : Same implementation as BlendSubtract
    return max(base + blend - vec3(1.), vec3(0.));
}

vec3 blendLinearBurn(in vec3 base, in vec3 blend, in float opacity) {
    return (blendLinearBurn(base, blend) * opacity + base * (1. - opacity));
}
#endif
