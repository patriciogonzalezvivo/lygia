import lygia::math::saturateMediump::saturateMediump;

// Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
fn smithGGXCorrelated(NoV: f32, NoL: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness;
    // TODO: lambdaV can be pre-computed for all the lights, it should be moved out of this function
    let lambdaV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
    let lambdaL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
    let v = 0.5 / (lambdaV + lambdaL);
    // a2=0 => v = 1 / 4*NoL*NoV   => min=1/4, max=+inf
    // a2=1 => v = 1 / 2*(NoL+NoV) => min=1/4, max=+inf
    // clamp to the maximum value representable in mediump
    return saturateMediump(v);
}

// Hammon 2017, "PBR Diffuse Lighting for GGX+Smith Microsurfaces"
fn smithGGXCorrelated_Fast(NoV: f32, NoL: f32, roughness: f32) -> f32 {
    let v = 0.5 / mix(2.0 * NoL * NoV, NoL + NoV, roughness);
    return saturateMediump(v);
}
