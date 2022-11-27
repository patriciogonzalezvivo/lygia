
#include "../../math/saturateMediump.glsl"

#ifndef FNC_KELEMEN
#define FNC_KELEMEN

// Kelemen 2001, "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
float kelemen(const in float LoH) {
    return saturateMediump(0.25 / (LoH * LoH));
}

#endif