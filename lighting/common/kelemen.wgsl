#include "../../math/saturateMediump.wgsl"

// Kelemen 2001, "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
fn kelemen(LoH: f32) -> f32 {
    return saturateMediump(0.25 / (LoH * LoH));
}
