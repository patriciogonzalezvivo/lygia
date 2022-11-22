
#include "../../math/saturateMediump.hlsl"

#ifndef FNC_KELEMEN
#define FNC_KELEMEN

// Kelemen 2001, "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
float kelemen(float LoH) {
    return saturateMediump(0.25 / (LoH * LoH));
}

#endif