/*
original_author: Jamie Owen
description: Photoshop Average blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendAverage(<float|float3> base, <float|float3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDAVERAGE
#define FNC_BLENDAVERAGE
float blendAverage(in float base, in float blend) {
    return (base + blend) * .5;
}

float3 blendAverage(in float3 base, in float3 blend) {
    return (base + blend) * .5;
}

float3 blendAverage(in float3 base, in float3 blend, float opacity) {
    return (blendAverage(base, blend) * opacity + base * (1. - opacity));
}
#endif
