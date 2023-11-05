
#include "../luminance.hlsl"

/*
contributors: [Erik Reinhard, Michael Stark, Peter Shirley, James Ferwerda]
description: Photographic Tone Reproduction for Digital Images. http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/hdr_photographic.pdf
use: <float3|float4> tonemapReinhardJodie(<float3|float4> x)
*/

#ifndef FNC_TONEMAPREINHARDJODIE
#define FNC_TONEMAPREINHARDJODIE
float3 tonemapReinhardJodie(const float3 x) { 
    float l = luminance(x);
    float3 tc = x / (x + 1.0);
    return lerp(x / (l + 1.0), tc, tc); 
}
float4 tonemapReinhardJodie(const float4 x) { return float4( tonemapReinhardJodie(x.rgb), x.a ); }
#endif