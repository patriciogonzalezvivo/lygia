#ifndef FNC_GTAOMULTIBOUNCE
#define FNC_GTAOMULTIBOUNCE

/*
contributors: NAN
description: |
    Returns a color ambient occlusion based on a pre-computed visibility term.
    The albedo term is meant to be the diffuse color or f0 for the diffuse and
    specular terms respectively.
use: <float3> gtaoMultiBounce(<float> visibility, <float3> albedo)
*/

float3 gtaoMultiBounce(float visibility, const float3 albedo) {
    // Jimenez et al. 2016, "Practical Realtime Strategies for Accurate Indirect Occlusion"
    float3 a =  2.0404 * albedo - 0.3324;
    float3 b = -4.7951 * albedo + 0.6417;
    float3 c =  2.7552 * albedo + 0.6903;

    return max(float3(visibility, visibility, visibility), ((visibility * a + b) * visibility + c) * visibility);
}

#endif