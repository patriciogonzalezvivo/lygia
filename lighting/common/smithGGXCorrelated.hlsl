
#include "../../math/saturateMediump.hlsl"

#ifndef FNC_SMITH_GGX_CORRELATED
#define FNC_SMITH_GGX_CORRELATED

float smithGGXCorrelated(float NoV, float NoL, float roughness) {
    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    float a2 = roughness * roughness;
    // TODO: lambdaV can be pre-computed for all the lights, it should be moved out of this function
    float lambdaV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
    float lambdaL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
    float v = 0.5 / (lambdaV + lambdaL);
    // a2=0 => v = 1 / 4*NoL*NoV   => min=1/4, max=+inf
    // a2=1 => v = 1 / 2*(NoL+NoV) => min=1/4, max=+inf
    // clamp to the maximum value representable in mediump
    return saturateMediump(v);
}

float smithGGXCorrelated_Fast(float NoV, float NoL, float roughness) {
    // Hammon 2017, "PBR Diffuse Lighting for GGX+Smith Microsurfaces"
    float v = 0.5 / lerp(2.0 * NoL * NoV, NoL + NoV, roughness);
    return saturateMediump(v);
}

#endif