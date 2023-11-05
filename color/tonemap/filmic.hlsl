

#include "../luminance.hlsl"

/*
contributors: [Jim Hejl, Richard Burgess-Dawson ]
description: Haarm-Peter Duikers curve from John Hables presentation "Uncharted 2 HDR Lighting", Page 140, http://www.gdcvault.com/play/1012459/Uncharted_2__HDR_Lighting
use: <float3|float4> tonemapFilmic(<float3|float4> x)
*/

#ifndef FNC_TONEMAPFILMIC
#define FNC_TONEMAPFILMIC
float3 tonemapFilmic(float3 color) {
    color = max(float3(0.0, 0.0, 0.0), color - 0.004);
    color = (color * (6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06);
    return color;
}

float4 tonemapFilmic(const float4 x) { return float4( tonemapFilmic(x.rgb), x.a ); }
#endif