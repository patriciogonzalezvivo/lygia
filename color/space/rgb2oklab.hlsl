#include "rgb2srgb.hlsl"
#include "srgb2oklab.hlsl"

/*
contributors: Bjorn Ottosson (@bjornornorn)
description: |
    Linear rgb ot OKLab https://bottosson.github.io/posts/oklab/
use: <float3\float4> rgb2oklab(<float3|float4> srgb)
*/

#ifndef FNC_RGB2OKLAB
#define FNC_RGB2OKLAB
float3 rgb2oklab(const in float3 rgb) { srgb2oklab( rgb2srgb(rgb) ); }
float4 rgb2oklab(const in float4 rgb) { return float4(rgb2oklab(rgb.rgb), rgb.a); }
#endif