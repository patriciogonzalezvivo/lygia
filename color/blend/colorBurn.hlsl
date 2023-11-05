/*
contributors: Jamie Owen
description: Photoshop Color Burn blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendColorBurn(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDCOLORBURN
#define FNC_BLENDCOLORBURN
float blendColorBurn(in float base, in float blend) {
    return (blend == 0.)? blend: max((1. - ((1. - base ) / blend)), 0.);
}

float3 blendColorBurn(in float3 base, in float3 blend) {
    return float3(  blendColorBurn(base.r, blend.r),
                    blendColorBurn(base.g, blend.g),
                    blendColorBurn(base.b, blend.b));
}

float3 blendColorBurn(in float3 base, in float3 blend, in float opacity) {
    return (blendColorBurn(base, blend) * opacity + base * (1. - opacity));
}
#endif
