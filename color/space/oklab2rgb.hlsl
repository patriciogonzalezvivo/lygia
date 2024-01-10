#include "oklab2srgb.hlsl"
#include "srgb2rgb.hlsl"

/*
contributors: Bjorn Ottosson (@bjornornorn)
description: oklab to linear RGB https://bottosson.github.io/posts/oklab/
use: <float3\float4> oklab2rgb(<float3|float4> oklab)
*/

#ifndef FNC_OKLAB2RGB
#define FNC_OKLAB2RGB
float3 oklab2rgb(const in float3 oklab) { return srgb2rgb(oklab2srgb(oklab)); }
float4 oklab2rgb(const in float4 oklab) { return float4(oklab2rgb(oklab.xyz), oklab.a); }
#endif