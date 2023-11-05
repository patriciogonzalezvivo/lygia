#include "../luminance.hlsl"

/*
contributors: [Erik Reinhard, Michael Stark, Peter Shirley, James Ferwerda]
description: Photographic Tone Reproduction for Digital Images. http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/hdr_photographic.pdf
use: <float3|float4> tonemapReinhard(<float3|float4> x)
*/

#ifndef FNC_TONEMAPREINHARD
#define FNC_TONEMAPREINHARD
float3 tonemapReinhard(const float3 x) { return x / (1.0 + luminance(x)); }
float4 tonemapReinhard(const float4 x) { return float4( tonemapReinhard(x.rgb), x.a ); }
#endif