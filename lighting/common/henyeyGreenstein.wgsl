#include "../../math/const.wgsl"

// Henyey-Greenstein phase function factor [-1, 1]
// represents the average cosine of the scattered directions
// 0 is isotropic scattering
// > 1 is forward scattering, < 1 is backwards
const HENYEYGREENSTEIN_SCATTERING: f32 = 0.76;

fn henyeyGreenstein(mu: f32) -> f32 {
    return max(0.0, (1.0 - HENYEYGREENSTEIN_SCATTERING*HENYEYGREENSTEIN_SCATTERING) / ((4. + PI) * pow(1.0 + HENYEYGREENSTEIN_SCATTERING*HENYEYGREENSTEIN_SCATTERING - 2.0 * HENYEYGREENSTEIN_SCATTERING * mu, 1.5)));
}

fn henyeyGreensteina(mu: f32, g: f32) -> f32 {
    let gg = g * g;
    return (1.0 / (4.0 * PI)) * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

fn henyeyGreensteinb(mu: f32, g: f32, dual_lobe_weight: f32) -> f32 {
    return mix(henyeyGreenstein( mu, -g), henyeyGreenstein(mu, g), dual_lobe_weight);
}
