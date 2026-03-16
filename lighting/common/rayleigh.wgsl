#include "../../math/const.wgsl"

// Rayleigh phase
fn rayleigh(mu: f32) -> f32 {
    return 3. * (1. + mu*mu) / (16. * PI);
}
