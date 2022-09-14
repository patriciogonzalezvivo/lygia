#include "lighten.hlsl"
#include "darken.hlsl"

/*
original_author: Jamie Owen
description: Photoshop Pin Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendPinLight(<float|float3> base, <float|float3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDPINLIGHT
#define FNC_BLENDPINLIGHT
float blendPinLight(in float base, in float blend) {
    return (blend < .5)? blendDarken(base, (2.*blend)): blendLighten(base, (2. * (blend - .5)));
}

float3 blendPinLight(in float3 base, in float3 blend) {
    return float3(    blendPinLight(base.r, blend.r),
                      blendPinLight(base.g, blend.g),
                      blendPinLight(base.b, blend.b) );
}

float3 blendPinLight(in float3 base, in float3 blend, in float opacity) {
    return (blendPinLight(base, blend) * opacity + base * (1. - opacity));
}
#endif
