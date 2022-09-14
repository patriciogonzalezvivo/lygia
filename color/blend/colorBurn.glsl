/*
original_author: Jamie Owen
description: Photoshop Color Burn blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendColorBurn(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDCOLORBURN
#define FNC_BLENDCOLORBURN
float blendColorBurn(in float base, in float blend) {
    return (blend == 0.)? blend: max((1. - ((1. - base ) / blend)), 0.);
}

vec3 blendColorBurn(in vec3 base, in vec3 blend) {
    return vec3(blendColorBurn(base.r, blend.r),
                blendColorBurn(base.g, blend.g),
                blendColorBurn(base.b, blend.b));
}

vec3 blendColorBurn(in vec3 base, in vec3 blend, in float opacity) {
    return (blendColorBurn(base, blend) * opacity + base * (1. - opacity));
}
#endif
